#!/bin/bash
# After the AWS fingerprinting run: relaunch any unfinished zips (up to 2 times), then start the audit.
#
#   nohup caffeinate -i scripts/aws_after_fp.sh > data/aws_after_fp.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
TOTAL=186
RETRIES=0
while true; do
  RUNNING=$(aws ec2 describe-instances --region us-east-1 --filters Name=tag:Name,Values=provena-fp \
            Name=instance-state-name,Values=pending,running --query "length(Reservations[].Instances[])" --output text)
  DONE=$(aws s3 ls s3://provena-openvid-696865460743/done/ 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date '+%F %T') fingerprint machines running=$RUNNING zips done=$DONE/$TOTAL"
  if [ "$RUNNING" = 0 ]; then
    if [ "$DONE" -ge "$TOTAL" ]; then
      QUOTA=$(aws service-quotas get-service-quota --region us-east-1 --service-code ec2 --quota-code L-34B43A08 \
              --query Quota.Value --output text | cut -d. -f1)
      N=$(( QUOTA / 8 < 50 ? QUOTA / 8 : 50 ))
      echo "all zips done; launching the audit on $N machines"
      scripts/aws_openvid.sh audit "$N"
      exit
    fi
    if [ "$RETRIES" -ge 2 ]; then echo "zips still missing after 2 relaunches; stopping"; exit 1; fi
    RETRIES=$((RETRIES + 1))
    echo "relaunching unfinished zips (attempt $RETRIES)"
    scripts/aws_openvid.sh launch $(( (TOTAL - DONE) < 24 ? (TOTAL - DONE) : 24 ))
  fi
  sleep 600
done
