#!/bin/bash
# Fingerprint OpenVid-1M zips on AWS spot machines (Graviton, 4 CPUs), shards straight to S3.
#
#   scripts/aws_openvid.sh setup                 # bucket, machine role, upload the package + runner
#   scripts/aws_openvid.sh launch N [ZIP ...]    # N spot machines share the zips (default: every zip not done)
#   scripts/aws_openvid.sh status                # running machines, finished zips
#
# Each machine installs provena-audit, runs scripts/openvid_fingerprint.py on its zips (i, i+N, ...), writes
# s3://BUCKET/done/<zip> after each one, uploads its log to s3://BUCKET/logs/ and terminates itself.
# Run from the repo root.
set -euo pipefail
REGION=us-east-1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=provena-openvid-$ACCOUNT
ROLE=provena-ec2
TYPE=${TYPE:-c7g.xlarge}
LIMIT=${LIMIT:-0}
OUT=${OUT:-openvid_send}

case "$1" in
setup)
  aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null || aws s3api create-bucket --bucket "$BUCKET" --region $REGION >/dev/null
  if ! aws iam get-role --role-name $ROLE >/dev/null 2>&1; then
    aws iam create-role --role-name $ROLE --assume-role-policy-document \
      '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
    aws iam create-instance-profile --instance-profile-name $ROLE >/dev/null
    aws iam add-role-to-instance-profile --instance-profile-name $ROLE --role-name $ROLE
  fi
  aws iam put-role-policy --role-name $ROLE --policy-name bucket --policy-document \
    "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"s3:*\",\"Resource\":[\"arn:aws:s3:::$BUCKET\",\"arn:aws:s3:::$BUCKET/*\"]}]}"
  tar czf /tmp/provena-audit.tgz --exclude .venv --exclude build --exclude dist --exclude __pycache__ provena-audit
  aws s3 cp /tmp/provena-audit.tgz "s3://$BUCKET/code/provena-audit.tgz" --quiet
  aws s3 cp scripts/openvid_fingerprint.py "s3://$BUCKET/code/openvid_fingerprint.py" --quiet
  echo "ready: s3://$BUCKET"
  ;;
launch)
  N=$2; shift 2
  if [ $# -gt 0 ]; then ZIPS=("$@"); else
    DONE=$(aws s3 ls "s3://$BUCKET/done/" 2>/dev/null | awk '{print $4}' || true)
    ZIPS=($(python3 - "$DONE" <<'EOF'
import json, re, sys, urllib.request
done = set(sys.argv[1].split())
files = [f["path"] for f in json.load(urllib.request.urlopen(
    "https://huggingface.co/api/datasets/nkp37/OpenVid-1M/tree/main")) if f["type"] == "file"]
zips = sorted({f if f.endswith(".zip") else re.sub(r"_part[a-z]{2}$", "", f) for f in files
               if re.match(r"OpenVid_part\d+(\.zip|_part[a-z]{2})$", f)})
print(" ".join(z for z in zips if z not in done))
EOF
))
  fi
  AMI=$(aws ssm get-parameter --region $REGION --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64 \
        --query Parameter.Value --output text)
  echo "${#ZIPS[@]} zips on $N machines ($TYPE spot)"
  for ((i = 0; i < N; i++)); do
    MINE=""
    for ((j = i; j < ${#ZIPS[@]}; j += N)); do MINE="$MINE ${ZIPS[$j]}"; done
    [ -z "$MINE" ] && continue
    USERDATA=$(cat <<EOF
#!/bin/bash
exec > /var/log/fp.log 2>&1
dnf install -y python3.12 python3.12-devel gcc-c++
python3.12 -m venv /opt/v
/opt/v/bin/pip install -q requests stream-unzip s3fs
aws s3 cp s3://$BUCKET/code/provena-audit.tgz /tmp/ && tar xzf /tmp/provena-audit.tgz -C /tmp
/opt/v/bin/pip install -q /tmp/provena-audit
aws s3 cp s3://$BUCKET/code/openvid_fingerprint.py /opt/
ID=\$(cat /var/lib/cloud/data/instance-id)
for z in $MINE; do
  aws s3 ls s3://$BUCKET/done/\$z >/dev/null && continue
  /opt/v/bin/python /opt/openvid_fingerprint.py \$z --out s3://$BUCKET/$OUT --workers 4 --limit $LIMIT \
    && [ $LIMIT = 0 ] && echo ok | aws s3 cp - s3://$BUCKET/done/\$z  # timing runs (LIMIT) don't count as done
  aws s3 cp /var/log/fp.log s3://$BUCKET/logs/\$ID.log --quiet
done
aws s3 cp /var/log/fp.log s3://$BUCKET/logs/\$ID.log --quiet
shutdown -h now
EOF
)
    aws ec2 run-instances --region $REGION --image-id "$AMI" --instance-type "$TYPE" \
      --iam-instance-profile Name=$ROLE --instance-initiated-shutdown-behavior terminate \
      --instance-market-options '{"MarketType":"spot"}' \
      --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
      --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=provena-fp}]' \
      --user-data "$USERDATA" --query "Instances[0].InstanceId" --output text
  done
  ;;
status)
  aws ec2 describe-instances --region $REGION --filters Name=tag:Name,Values=provena-fp \
    Name=instance-state-name,Values=pending,running --query "Reservations[].Instances[].[InstanceId,State.Name,LaunchTime]" --output text
  echo "zips done: $(aws s3 ls "s3://$BUCKET/done/" 2>/dev/null | wc -l | tr -d ' ')"
  ;;
esac
