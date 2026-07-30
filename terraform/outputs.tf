###############################################################################
# outputs.tf — gpu-power-lab Terraform stack
###############################################################################

output "instance_id" {
  description = "EC2 instance ID."
  value       = aws_instance.gpu_lab.id
}

output "public_ip" {
  description = "Public IPv4 address of the GPU instance."
  value       = aws_instance.gpu_lab.public_ip
}

output "public_dns" {
  description = "Public DNS name of the GPU instance."
  value       = aws_instance.gpu_lab.public_dns
}

output "ssh_command" {
  description = <<-EOT
    Convenience SSH command. Replace <keyfile> with the path to your
    private key (the counterpart of ssh_public_key / key_name).
  EOT
  value = "ssh -i <keyfile> ubuntu@${aws_instance.gpu_lab.public_dns}"
}

output "results_bucket_arn" {
  description = "ARN of the S3 results bucket (data source, not created here)."
  value       = data.aws_s3_bucket.results.arn
}

output "ami_id_resolved" {
  description = "AMI ID that was actually used for the instance."
  value       = aws_instance.gpu_lab.ami
}

output "iam_role_arn" {
  description = "ARN of the IAM role attached to the instance."
  value       = aws_iam_role.gpu_lab.arn
}
