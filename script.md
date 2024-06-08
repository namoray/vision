Subnet 19 is a Bittensor subnet, focused on the decentralised inference of AI models. 

Here we call ourselves Vision 👀. Bittensor has the vision of being the most powerful force in AI, powering the world through decentralisation - SN19 is all about fulfilling that Vision!

Vision works by entitites known as validators, incentivising entites known as miners. Miners run AI models such as llama3 or stable diffusion, and validators check if they're doing a good job 👍

Here's a deep dive on the actual mechanism as play:

The subnet is split into tasks - again, such as llama3 or stable diffusion.


Each of these independent tasks have a weighting or 'importance' associated with them for example 60% and 40%. Tasks are competely independent of eachother.

For each task a miner decides on a specific capacity they can handle for task, per hour, and tell the validators

Validators take the announced capacity, and randomly decide how much of that capacity they will check over the next hour. 

After an hour of sending queries validators calculate a `period score` for each miner. 

This score reflects the percentage of requests that were correctly responded to. Miners are penalized for missed queries based on the volume left unqueried. If the announced volume is fully utilized, no penalties are imposed. However, if a miner is queried only once or twice and fails to respond, the penalties are more severe.

Simultaneously, validators assess the quality of all responses from miners, maintaining a running tally of their 'quality scores' and 'speed factors' (rewarding both speed and correctness). When it is time to set weights, validators calculate a volume-weighted average period score for each miner. This score is then multiplied by the miner's combined quality score to derive an overall score for that specific task.

Finally, the overall scores for each miner are combined across all tasks - which is what makes the tasks completely optional for miners. The different scores for all tasks are then summed up to derive a final overall score for each miner, which is used to set weights appropriately.

Phew!

Ok that's a lot of text, what does it mean?
It means a few things:

Miners can run tasks completely optionally. The more tasks they run, with greater volumes and speeds, the more incentive they will get.
By controlling the reward distribution to miners, we can directly incentivise greater volumes and speeds for tasks that get greater organic usage. We could even give power for validators to have a stake weighted 'vote' for the tasks they care about...
We can add as many tasks as we like! If there's demand for something we can add it! If not, we can remove it! No registrations or deregistrations needed, miners can just scale up and scale down their capacity as needed.
Miners have the ability to rate limit explicitly to validators without incurring a greater penalty. This means we can much more effectively load balance between miners, to make sure any organic requests can be always handled by a miner who is capable of performing that task!
There's nothing special about a synthetic or organic query. No distinction can be made on the miner side, but validators can still give preferential treatment to organics!