#!/bin/bash
# Fingerprint Hugging Face video datasets on AWS spot machines (runner: scripts/hf_fingerprint.py).
#
#   scripts/aws_hf.sh plan REPO [REPO ...]     # list every zip/tar/parquet/video file of the datasets -> plan.txt
#   scripts/aws_hf.sh launch N                 # N machines share plan.txt; shards -> s3://BUCKET/index/<repo>/send/
#   scripts/aws_hf.sh status
#
# Gated datasets: store a Hugging Face read token once, in your own terminal (not in chat):
#   aws ssm put-parameter --region us-east-1 --name /provena/hf_token --type SecureString --value hf_...
# Machines read it at start-up. Uses the bucket and role from scripts/aws_openvid.sh setup. Run from the repo root.
set -euo pipefail
REGION=us-east-1
export AWS_DEFAULT_REGION=$REGION
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=provena-openvid-$ACCOUNT
ROLE=provena-ec2
TYPE=${TYPE:-c7g.2xlarge}
PLAN=${PLAN:-data/discovery/plan.txt}

case "$1" in
plan)
  shift
  python3 - "$@" > $PLAN <<'EOF'
import json, sys, urllib.request
EXT = (".zip", ".tar", ".tar.gz", ".tgz", ".parquet", ".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v")
lines = []
for repo in sys.argv[1:]:
    info = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}?blobs=true", timeout=60))
    files = [(s.get("size") or 0, s["rfilename"]) for s in info["siblings"]
             if s["rfilename"].lower().endswith(EXT) and "caption" not in s["rfilename"].lower()]
    lines += [(size, repo, f) for size, f in files]
lines.sort(key=lambda t: -t[0])  # biggest first, so machines (taking every Nth line) get similar loads
for _, repo, f in lines:
    print(f"{repo}\t{f}")
EOF
  echo "$(wc -l < $PLAN) files in $PLAN"; cut -f1 $PLAN | sort | uniq -c
  ;;
launch)
  N=$2
  aws iam put-role-policy --role-name $ROLE --policy-name hf-token --policy-document \
    "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"ssm:GetParameter\",\"Resource\":\"arn:aws:ssm:$REGION:$ACCOUNT:parameter/provena/hf_token\"}]}"
  tar czf /tmp/provena-audit.tgz --exclude .venv --exclude build --exclude dist --exclude __pycache__ provena-audit
  aws s3 cp /tmp/provena-audit.tgz "s3://$BUCKET/code/provena-audit.tgz" --quiet
  aws s3 cp scripts/hf_fingerprint.py "s3://$BUCKET/code/hf_fingerprint.py" --quiet
  aws s3 cp $PLAN "s3://$BUCKET/code/plan.txt" --quiet
  AMI=$(aws ssm get-parameter --region $REGION --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64 \
        --query Parameter.Value --output text)
  echo "$(wc -l < $PLAN) files on $N machines ($TYPE spot)"
  for ((i = 0; i < N; i++)); do
    USERDATA=$(cat <<EOF
#!/bin/bash
exec > /var/log/fp.log 2>&1
dnf install -y python3.12 python3.12-devel gcc-c++
python3.12 -m venv /opt/v
/opt/v/bin/pip install -q requests stream-unzip s3fs pyarrow
aws s3 cp s3://$BUCKET/code/provena-audit.tgz /tmp/ && tar xzf /tmp/provena-audit.tgz -C /tmp
/opt/v/bin/pip install -q /tmp/provena-audit
aws s3 cp s3://$BUCKET/code/hf_fingerprint.py /opt/ && aws s3 cp s3://$BUCKET/code/plan.txt /opt/
export HF_TOKEN=\$(aws ssm get-parameter --region $REGION --name /provena/hf_token --with-decryption \
  --query Parameter.Value --output text 2>/dev/null || true)
ID=\$(cat /var/lib/cloud/data/instance-id)
( while true; do sleep 300; tail -n 3 /var/log/fp_*.log > /tmp/progress.txt; aws s3 cp /tmp/progress.txt s3://$BUCKET/logs/hf_\$ID.log --quiet; done ) &
for k in 0 1 2 3 4 5 6 7; do  # 8 files in flight per machine, one per core (long single videos use one core each)
  /opt/v/bin/python /opt/hf_fingerprint.py --plan /opt/plan.txt --part \$(( $i * 8 + k )) --of $(( N * 8 )) \
    --out s3://$BUCKET/index --workers 1 > /var/log/fp_\$k.log 2>&1 &
done
wait
cat /var/log/fp_*.log >> /var/log/fp.log
aws s3 cp /var/log/fp.log s3://$BUCKET/logs/hf_\$ID.log --quiet
shutdown -h now
EOF
)
    aws ec2 run-instances --region $REGION --image-id "$AMI" --instance-type "$TYPE" \
      --iam-instance-profile Name=$ROLE --instance-initiated-shutdown-behavior terminate \
      --instance-market-options '{"MarketType":"spot"}' \
      --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
      --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=provena-hf}]' \
      --user-data "$USERDATA" --query "Instances[0].InstanceId" --output text
  done
  ;;
status)
  echo "machines running: $(aws ec2 describe-instances --region $REGION --filters Name=tag:Name,Values=provena-hf \
    Name=instance-state-name,Values=pending,running --query 'length(Reservations[].Instances[])' --output text)"
  for d in $(aws s3 ls "s3://$BUCKET/index/_done/" | awk '{print $2}'); do
    echo "  ${d%/}: $(aws s3 ls "s3://$BUCKET/index/_done/$d" | wc -l | tr -d ' ') files done"
  done
  echo "plan: $(wc -l < $PLAN) files"
  ;;
esac
