###############################################################################
# variables.tf — gpu-power-lab Terraform stack
###############################################################################

variable "aws_region" {
  type        = string
  default     = "ap-southeast-2"
  nullable    = false
  description = "AWS region in which to create all resources."
}

variable "ssh_ingress_cidr" {
  type        = string
  nullable    = false
  description = <<-EOT
    CIDR block allowed to reach port 22 on the GPU instance.
    No default — you must provide your own IP, e.g. "203.0.113.42/32".
    Using "0.0.0.0/0" opens SSH to the internet; use only in a test context.
  EOT

  validation {
    condition     = can(cidrnetmask(var.ssh_ingress_cidr))
    error_message = "ssh_ingress_cidr must be a valid CIDR block, e.g. \"203.0.113.42/32\"."
  }
}

variable "ssh_public_key" {
  type        = string
  default     = ""
  description = <<-EOT
    OpenSSH public key material (the contents of id_rsa.pub or similar).
    Required when key_name is null — a new key pair will be created in AWS.
    Ignored when key_name is set to an existing key pair name.
  EOT
}

variable "key_name" {
  type        = string
  default     = null
  nullable    = true
  description = <<-EOT
    Name of an existing EC2 key pair to use. When set, ssh_public_key is
    not used and no new key pair resource is created. When null (default),
    a key pair is created from ssh_public_key.
  EOT
}

variable "instance_type" {
  type        = string
  default     = "g5.xlarge"
  nullable    = false
  description = <<-EOT
    EC2 instance type. GPU instances of interest:
      g5.xlarge   — A10G  (300 W)  ~$1.20/hr on-demand in ap-southeast-2
      g6e.xlarge  — L40S  (350 W)
      p4d.24xlarge — 8×A100 (400 W each)  ~$40/hr — be careful
      p5.48xlarge  — 8×H100 (700 W each)  ~$100/hr — destroy immediately after use
  EOT
}

variable "ami_id" {
  type        = string
  default     = null
  nullable    = true
  description = <<-EOT
    Override AMI ID. When null (default) the latest AWS Deep Learning AMI
    GPU PyTorch on Ubuntu 22.04 is resolved automatically via data source.
    Set this to pin a specific tested AMI version.
  EOT
}

variable "use_spot" {
  type        = bool
  default     = false
  nullable    = false
  description = <<-EOT
    When true, the instance is launched as a one-time Spot request via
    instance_market_options on aws_instance. Spot is ~70% cheaper but the
    instance can be interrupted with 2 minutes notice. Ensure campaign.py
    checkpointing is adequate before enabling this for long campaigns.
  EOT
}

variable "results_bucket" {
  type        = string
  nullable    = false
  description = <<-EOT
    Name (not ARN) of the existing S3 bucket for campaign results.
    The bucket must already exist — this stack does not create it.
    Example: "my-gpu-lab-results"
  EOT
}

variable "results_prefix" {
  type        = string
  default     = "gpu-power-lab/"
  nullable    = false
  description = <<-EOT
    S3 key prefix under which campaign results are stored.
    Must end with a slash. The IAM policy is scoped to this prefix.
    Example: "gpu-power-lab/campaigns/"
  EOT

  validation {
    condition     = endswith(var.results_prefix, "/")
    error_message = "results_prefix must end with a forward slash, e.g. \"gpu-power-lab/\"."
  }
}

variable "timestream_database" {
  type        = string
  default     = null
  nullable    = true
  description = <<-EOT
    Name of an existing Amazon Timestream database for telemetry writes.
    Set together with timestream_table to enable Timestream IAM policy.
    Leave null to skip Timestream entirely (default).
  EOT
}

variable "timestream_table" {
  type        = string
  default     = null
  nullable    = true
  description = <<-EOT
    Name of the Timestream table within timestream_database.
    Must be set together with timestream_database.
  EOT

  validation {
    condition     = (var.timestream_database == null) == (var.timestream_table == null)
    error_message = "timestream_database and timestream_table must both be set or both be null."
  }
}

variable "repo_url" {
  type        = string
  default     = ""
  description = <<-EOT
    Git URL of the gpu-power-lab repository to clone on first boot.
    Example: "https://github.com/yourorg/gpu-power-lab.git"
    When empty, the clone step is skipped and the operator is expected
    to transfer the source manually (e.g. scp or rsync).
  EOT
}

variable "repo_ref" {
  type        = string
  default     = "main"
  nullable    = false
  description = "Git ref (branch, tag, or commit SHA) to check out after cloning."
}
