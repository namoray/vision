# 4.0

## 🚀 High Level Flow Changes
-    Capacities announced by miners.
-    Validators now calculate their allowed capacity for each operation for each miner.
-    Validators synthetically score miners on these requests, calculating a `period_score`, which represents reliability.
-    Validators workflow for each task:
       - Use the quality scores calculated from the scoring of all tasks for this UID, with a preference for quality scores from this specific task.
        - Use a decaying weighted average of previous period scores to calculate a weighted period score.
        - Combine the weighted period score with the task quality score and multiply by volume to get the effective volume (capacity).
        - Compare the effective capacities with others to get a score for this UID for the task.
        - Weight all scores across each task using the subnet-level task importances to derive an overall incentive score.
-    Miners can rate limit explicitly to validators without incurring a greater penalty, depending on volume.

## 🆙 Upgrades
-    Bittensor upgrade to version 6.12.1.

## 🔧 Validator Enhancements
-    Validators to store extensive stats information.
-    Reworked weight setting to utilize announced capacities and 'evidence' of those capacities.
-    Increased range for speed scores.
-    Posted stats for sexy dashboards.
-    Small bug fixes for validator proxy.
-    Implemented smart load balancing between miners for organic requests using a doubly linked list.
-    Cleaned up outdated code.
-    Separated validator into specific duties.
