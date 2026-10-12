#!/bin/bash
# Wait for the EC2 spot vCPU quota to be raised, then launch the full OpenVid fingerprinting run.
#
#   nohup caffeinate -i scripts/aws_wait_and_launch.sh REQUEST_ID > aws_wait.log 2>&1 &
#
# Checks every 10 minutes. Launches min(100, quota / 4) machines once the quota is at least 32 vCPUs;
# stops if the request is denied or closed without a raise.
set -uo pipefail
cd "$(dirname "$0")/.."
REQUEST=$1
while true; do
  QUOTA=$(aws service-quotas get-service-quota --region us-east-1 --service-code ec2 --quota-code L-34B43A08 \
          --query Quota.Value --output text 2>/dev/null | cut -d. -f1)
  STATUS=$(aws service-quotas get-requested-service-quota-change --region us-east-1 --request-id "$REQUEST" \
           --query RequestedQuota.Status --output text 2>/dev/null)
  echo "$(date '+%F %T') quota=${QUOTA:-?} request=${STATUS:-?}"
  if [ -n "$QUOTA" ] && [ "$QUOTA" -ge 32 ]; then
    N=$(( QUOTA / 4 < 100 ? QUOTA / 4 : 100 ))
    echo "launching $N machines"
    scripts/aws_openvid.sh launch "$N"
    exit
  fi
  case "$STATUS" in DENIED|CASE_CLOSED|NOT_APPROVED) echo "request $STATUS, not launching"; exit 1 ;; esac
  sleep 600
done
