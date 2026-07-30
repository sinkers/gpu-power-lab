###############################################################################
# gpu-power-lab — Terraform stack
# Provisions a single GPU instance for building the C runner and running
# a benchmark campaign.
#
# Terraform ≥ 1.5 required.
###############################################################################

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0.0"
    }
  }
}

###############################################################################
# Provider
###############################################################################

provider "aws" {
  region = var.aws_region
}

###############################################################################
# VPC strategy: use the account's default VPC and its default subnets.
#
# Rationale: a self-contained VPC adds ~200 lines of boilerplate (IGW,
# route tables, NACL) for no practical benefit in a single-instance lab.
# The default VPC is present in every region, already has an IGW, and
# its default subnets auto-assign public IPs — exactly what we need.
# To migrate to a custom VPC, swap the two data sources below and supply
# a subnet_id variable.
###############################################################################

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    # Only grab the AZ-default subnets (they have map_public_ip_on_launch=true)
    name   = "defaultForAz"
    values = ["true"]
  }
}

###############################################################################
# Caller identity (used in IAM ARN construction)
###############################################################################

data "aws_caller_identity" "current" {}

###############################################################################
# AMI — Latest AWS Deep Learning AMI GPU PyTorch on Ubuntu 22.04
#
# Owner 898082745236 is the canonical Amazon-published DLAMI account.
# These AMIs ship with CUDA, cuDNN, cuBLAS, NVML, and a working driver
# stack, so user-data only needs to build the runner on top.
#
# If the owner ID ever changes, set var.ami_id to pin a specific AMI and
# the data source is bypassed (see the instance resource below).
###############################################################################

data "aws_ami" "gpu" {
  most_recent = true
  owners      = ["898082745236"] # Amazon Deep Learning AMIs

  filter {
    name   = "name"
    values = ["Deep Learning AMI GPU PyTorch*Ubuntu 22.04*"]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
  filter {
    name   = "state"
    values = ["available"]
  }
}

###############################################################################
# Existing S3 bucket — NOT created here; assumed to exist.
###############################################################################

data "aws_s3_bucket" "results" {
  bucket = var.results_bucket
}

###############################################################################
# Security group
###############################################################################

resource "aws_security_group" "gpu_lab" {
  name        = "gpu-power-lab-ssh"
  description = "Allow SSH from operator CIDR; all egress."
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH from operator"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_ingress_cidr]
  }

  egress {
    description = "All egress"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name      = "gpu-power-lab-ssh"
    Project   = "gpu-power-lab"
    ManagedBy = "terraform"
  }
}

###############################################################################
# EC2 Key pair — created from ssh_public_key unless key_name is supplied.
###############################################################################

resource "aws_key_pair" "gpu_lab" {
  # count = 0 when the caller already has a named key pair in AWS.
  count = var.key_name == null ? 1 : 0

  key_name   = "gpu-power-lab-${data.aws_caller_identity.current.account_id}"
  public_key = var.ssh_public_key

  tags = {
    Project   = "gpu-power-lab"
    ManagedBy = "terraform"
  }
}

###############################################################################
# IAM — instance role
###############################################################################

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    sid     = "AllowEC2Assume"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "gpu_lab" {
  name               = "gpu-power-lab-instance-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json

  tags = {
    Project   = "gpu-power-lab"
    ManagedBy = "terraform"
  }
}

# S3 policy — scoped to the configured bucket + prefix.
data "aws_iam_policy_document" "s3" {
  statement {
    sid     = "S3ObjectAccess"
    actions = ["s3:PutObject", "s3:GetObject"]
    resources = [
      "${data.aws_s3_bucket.results.arn}/${var.results_prefix}*",
    ]
  }

  statement {
    sid     = "S3ListBucket"
    actions = ["s3:ListBucket"]
    resources = [data.aws_s3_bucket.results.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["${var.results_prefix}*"]
    }
  }
}

resource "aws_iam_policy" "s3" {
  name        = "gpu-power-lab-s3"
  description = "S3 access for gpu-power-lab campaign results"
  policy      = data.aws_iam_policy_document.s3.json

  tags = {
    Project   = "gpu-power-lab"
    ManagedBy = "terraform"
  }
}

resource "aws_iam_role_policy_attachment" "s3" {
  role       = aws_iam_role.gpu_lab.name
  policy_arn = aws_iam_policy.s3.arn
}

# Timestream policy — only created when both database and table vars are set.
locals {
  timestream_enabled    = var.timestream_database != null && var.timestream_table != null
  timestream_db_name    = coalesce(var.timestream_database, "_placeholder_")
  timestream_table_name = coalesce(var.timestream_table, "_placeholder_")
}

data "aws_iam_policy_document" "timestream" {
  statement {
    sid     = "TimestreamWrite"
    actions = ["timestream:WriteRecords"]
    resources = [
      "arn:aws:timestream:${var.aws_region}:${data.aws_caller_identity.current.account_id}:database/${local.timestream_db_name}/table/${local.timestream_table_name}",
    ]
  }

  # DescribeEndpoints is required for the Timestream SDK to discover
  # the regional write endpoint; it has no resource scope.
  statement {
    sid       = "TimestreamDescribeEndpoints"
    actions   = ["timestream:DescribeEndpoints"]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "timestream" {
  count = local.timestream_enabled ? 1 : 0

  name        = "gpu-power-lab-timestream"
  description = "Timestream write access for gpu-power-lab"
  policy      = data.aws_iam_policy_document.timestream.json

  tags = {
    Project   = "gpu-power-lab"
    ManagedBy = "terraform"
  }
}

resource "aws_iam_role_policy_attachment" "timestream" {
  count = local.timestream_enabled ? 1 : 0

  role       = aws_iam_role.gpu_lab.name
  policy_arn = aws_iam_policy.timestream[0].arn
}

resource "aws_iam_instance_profile" "gpu_lab" {
  name = "gpu-power-lab-instance-profile"
  role = aws_iam_role.gpu_lab.name

  tags = {
    Project   = "gpu-power-lab"
    ManagedBy = "terraform"
  }
}

###############################################################################
# EC2 instance
#
# Spot support: when var.use_spot = true an instance_market_options block
# is injected via a dynamic block, requesting a one-time Spot instance.
# We use aws_instance (not aws_spot_instance_request) for two reasons:
#  1. aws_instance with instance_market_options is the modern idiom —
#     aws_spot_instance_request is a legacy resource and doesn't support
#     all current instance options.
#  2. outputs (public_ip, etc.) are immediately available from aws_instance
#     rather than requiring a separate data lookup after spot fulfillment.
#
# AMI override: set var.ami_id to skip the data source entirely.
###############################################################################

resource "aws_instance" "gpu_lab" {
  ami           = var.ami_id != null ? var.ami_id : data.aws_ami.gpu.id
  instance_type = var.instance_type

  subnet_id = data.aws_subnets.default.ids[0]

  vpc_security_group_ids = [aws_security_group.gpu_lab.id]
  iam_instance_profile   = aws_iam_instance_profile.gpu_lab.name

  key_name = var.key_name != null ? var.key_name : aws_key_pair.gpu_lab[0].key_name

  # Spot support — injected only when use_spot = true.
  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        # one-time: instance terminates when interrupted (no relaunch).
        # For a campaign that must run to completion, prefer persistent
        # combined with a checkpointing strategy in campaign.py.
        spot_instance_type             = "one-time"
        instance_interruption_behavior = "terminate"
      }
    }
  }

  root_block_device {
    volume_size           = 200
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true

    tags = {
      Name      = "gpu-power-lab-root"
      Project   = "gpu-power-lab"
      ManagedBy = "terraform"
    }
  }

  user_data = templatefile("${path.module}/user-data.sh.tftpl", {
    repo_url       = var.repo_url
    repo_ref       = var.repo_ref
    results_bucket = var.results_bucket
    results_prefix = var.results_prefix
  })

  # Replace instance on any user-data change so bootstrap always runs.
  user_data_replace_on_change = true

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required" # IMDSv2 only
    http_put_response_hop_limit = 1
  }

  tags = {
    Name      = "gpu-power-lab"
    Project   = "gpu-power-lab"
    ManagedBy = "terraform"
    Spot      = tostring(var.use_spot)
  }
}
