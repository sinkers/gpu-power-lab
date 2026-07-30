# gpu-power-lab — Terraform

Provisions a single AWS GPU instance for building the C runner and running a
benchmark campaign. The stack is intentionally flat (no modules) so every
resource is visible in one pass.

---

## ⚠️  Cost warning — read first

| Instance | GPU | On-demand (ap-southeast-2) | Notes |
|---|---|---|---|
| `g5.xlarge` | A10G 300 W | **~$1.20 / hr** | Default; smoke-test size |
| `g6e.xlarge` | L40S 350 W | check current pricing | |
| `p4d.24xlarge` | 8 × A100 40 GB | **~$40 / hr** | Destroy immediately after use |
| `p5.48xlarge` | 8 × H100 80 GB | **~$100 / hr** | Destroy immediately after use |

**A `p4d.24xlarge` left running overnight costs ~$960. A `p5.48xlarge` costs
~$2 400.** Set up a billing alarm in AWS and run `terraform destroy` when done.

Spot pricing reduces costs by ~70% but instances can be interrupted with
2 minutes' notice. Enable with `use_spot = true`.

---

## Prerequisites

| Tool | Minimum version |
|---|---|
| [Terraform](https://developer.hashicorp.com/terraform/install) | 1.5.0 |
| [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) | any recent v2 |
| AWS credentials | configured via env vars or `~/.aws/credentials` |

The S3 results bucket must exist before you apply. This stack does not create
it:

```bash
aws s3api create-bucket \
  --bucket my-gpu-lab-results-bucket \
  --region ap-southeast-2 \
  --create-bucket-configuration LocationConstraint=ap-southeast-2
```

---

## Quick start

### 1. Copy and edit the example vars file

```bash
cd terraform/
cp terraform.tfvars.example terraform.tfvars
$EDITOR terraform.tfvars
```

Mandatory fields you must fill in:

| Variable | Description |
|---|---|
| `ssh_ingress_cidr` | Your IP in CIDR notation, e.g. `"203.0.113.42/32"`. Run `curl -s https://checkip.amazonaws.com` to find it. |
| `ssh_public_key` | Contents of your `~/.ssh/id_ed25519.pub` (or similar). Not required if you set `key_name` to an existing AWS key pair name. |
| `results_bucket` | Name of your existing S3 bucket. |

### 2. Initialise Terraform

```bash
terraform init
```

Downloads the AWS provider (≥ 5.0) and sets up the working directory.

### 3. Preview the plan

```bash
terraform plan
```

Review what will be created. Key resources:
- `aws_instance.gpu_lab` — the GPU instance
- `aws_security_group.gpu_lab` — SSH-only ingress
- `aws_iam_role.gpu_lab` + `aws_iam_instance_profile.gpu_lab` — scoped S3 access
- `aws_key_pair.gpu_lab` — created only when `key_name` is null

### 4. Apply

```bash
terraform apply
```

Type `yes` at the prompt. The instance is running within ~60 seconds;
the user-data bootstrap (build + clone) takes a further 3–8 minutes
depending on instance type.

### 5. Check bootstrap progress

```bash
# Get the public DNS from Terraform output
terraform output -raw public_dns

# SSH in (substitute your actual private key path)
ssh -i ~/.ssh/id_ed25519 ubuntu@$(terraform output -raw public_dns)

# Tail the bootstrap log
tail -f /var/log/gpu-lab-bootstrap.log
```

---

## SSH access

Terraform prints a ready-made command:

```bash
terraform output ssh_command
# → ssh -i <keyfile> ubuntu@ec2-xxx-xxx.ap-southeast-2.compute.amazonaws.com
```

Replace `<keyfile>` with the path to your private key.

If you used `key_name` to reference an existing key pair, use the matching
private key. If Terraform created the key pair from `ssh_public_key`, use
the corresponding private key on your workstation.

---

## Transfer sources manually (when `repo_url` is empty)

```bash
rsync -av --exclude='.git' --exclude='runner/build' \
  /path/to/gpu-power-lab/ \
  ubuntu@$(terraform output -raw public_dns):/opt/gpu-power-lab/

# Then build on the instance
ssh -i ~/.ssh/id_ed25519 ubuntu@$(terraform output -raw public_dns) \
  "cd /opt/gpu-power-lab && bash scripts/build.sh"
```

---

## Run a campaign

The systemd service is installed **disabled** so the operator can verify the
build and customise the plan before firing off a campaign.

### Option A — one-shot via systemd

```bash
# Enable and start (runs once, then stops)
sudo systemctl start gpu-power-lab-campaign.service

# Follow live output
sudo journalctl -f -u gpu-power-lab-campaign.service

# Check exit status
sudo systemctl status gpu-power-lab-campaign.service
```

### Option B — run directly (useful for ad-hoc plan files)

```bash
cd /opt/gpu-power-lab
source .venv/bin/activate 2>/dev/null || true

python3 orchestrator/campaign.py \
  --plan orchestrator/plans/example.yaml \
  --out-dir /var/lib/gpu-lab/out
```

### Customise the plan

Edit or copy `orchestrator/plans/example.yaml` before starting:

```bash
cp /opt/gpu-power-lab/orchestrator/plans/example.yaml \
   /opt/gpu-power-lab/orchestrator/plans/my-plan.yaml
$EDITOR /opt/gpu-power-lab/orchestrator/plans/my-plan.yaml
```

Update the `ExecStart` line in the service file, or just invoke Python
directly as shown in Option B.

### Retrieve results from S3

Results are uploaded to `s3://<results_bucket>/<results_prefix>` by
`orchestrator/upload.py` at the end of each campaign.

```bash
aws s3 sync \
  s3://my-gpu-lab-results-bucket/gpu-power-lab/ \
  ~/gpu-lab-results/
```

---

## Variables reference

See [`variables.tf`](variables.tf) for full descriptions and validation rules.

| Variable | Default | Required | Description |
|---|---|---|---|
| `aws_region` | `ap-southeast-2` | no | AWS region |
| `ssh_ingress_cidr` | — | **yes** | CIDR for SSH access |
| `ssh_public_key` | `""` | when `key_name` is null | Public key material |
| `key_name` | `null` | no | Existing EC2 key pair name |
| `instance_type` | `g5.xlarge` | no | EC2 instance type |
| `ami_id` | `null` | no | Pin a specific AMI; else resolved automatically |
| `use_spot` | `false` | no | Enable Spot pricing |
| `results_bucket` | — | **yes** | S3 bucket name (must exist) |
| `results_prefix` | `gpu-power-lab/` | no | S3 key prefix (must end with `/`) |
| `timestream_database` | `null` | no | Timestream DB name (optional) |
| `timestream_table` | `null` | no | Timestream table name (optional) |
| `repo_url` | `""` | no | Git URL to clone on first boot |
| `repo_ref` | `main` | no | Git ref to check out |

---

## Outputs

```bash
terraform output                   # all outputs
terraform output -raw public_ip    # raw value, no quotes
terraform output -raw ssh_command  # paste-ready SSH command
```

| Output | Description |
|---|---|
| `instance_id` | EC2 instance ID |
| `public_ip` | Public IPv4 address |
| `public_dns` | Public DNS name |
| `ssh_command` | Convenience SSH string (replace `<keyfile>`) |
| `results_bucket_arn` | ARN of the S3 results bucket |
| `ami_id_resolved` | Actual AMI used (useful when var.ami_id is null) |
| `iam_role_arn` | ARN of the instance IAM role |

---

## Spot instances

```hcl
# terraform.tfvars
use_spot = true
```

The instance uses `instance_market_options` with `market_type = "spot"` and
`spot_instance_type = "one-time"`. If interrupted, it terminates (not
hibernated or stopped). Outputs work normally — Terraform waits for the Spot
request to be fulfilled before proceeding.

For campaigns that must run to completion, prefer `use_spot = false` or
implement checkpointing in `campaign.py` before enabling Spot.

---

## Destroy

```bash
terraform destroy
```

This terminates the instance and deletes the security group, IAM role, and
(if created) the EC2 key pair. **The S3 bucket and its contents are not
touched** — it was created outside this stack.

Verify the instance is gone:

```bash
aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=gpu-power-lab" \
            "Name=instance-state-name,Values=running,pending,stopping" \
  --query 'Reservations[].Instances[].InstanceId' \
  --output text
# Should return empty
```

---

## Design notes

- **Default VPC**: uses `data "aws_vpc" { default = true }` + `data
  "aws_subnets"`. No custom VPC is created. This keeps the stack at ~250 lines
  while providing a fully routable public IP. To migrate to a custom VPC,
  replace the two data sources with an `aws_subnet` data lookup and supply the
  subnet ID.

- **AMI**: resolved via `data "aws_ami"` with owner `898082745236` (Amazon
  DLAMI account) and name glob `Deep Learning AMI GPU PyTorch*Ubuntu 22.04*`.
  The DLAMI ships with CUDA ≥ 12, cuBLAS, NVML, and a tested driver — no
  driver installation needed in user-data.

- **Spot mechanism**: `instance_market_options` block on `aws_instance`
  (modern path) rather than the legacy `aws_spot_instance_request` resource.
  This keeps outputs (public IP etc.) on a single resource and avoids the
  quirky `wait_for_fulfillment` lifecycle of the legacy resource.

- **S3 bucket**: not created by this stack (`data "aws_s3_bucket"` only) to
  prevent accidental bucket destruction on `terraform destroy`.
