# CogniRad Cross-Channel Energy and Reallocation Spec

## Goal

CogniRad should behave like a cognitive-radio messaging system, not a fixed room chat.

The target behavior is:

- Students may send messages to students on the same channel.
- Students may also send messages across channels.
- Every message increases energy at three levels:
  - message energy
  - per-student cumulative energy
  - per-channel cumulative energy
- Channel energy must not only increase forever.
- Energy must decay downward during idle periods.
- When a channel crosses its safe energy threshold, the system should reduce congestion by moving students to other channels with spare capacity.
- Reallocation should be fair, so the same student is not always moved first.
- Round-robin logic should help distribute pressure fairly across channels and across candidate students.

## Corrected System Model

### 1. Channel meaning

A channel is a shared spectrum band with its own:

- current members
- total energy
- status
- safe capacity threshold

It is not just a chat room.

### 2. Messaging meaning

Messaging should support:

- same-channel messaging
- cross-channel messaging

Cross-channel messaging is allowed, but the energy side effects are still tracked against the sender and the sender's current channel occupancy/load model.

If later we want radio-accurate inter-channel cost, we can add an extra cross-channel routing penalty, but that is optional for the next version.

## Energy Model

The system should track energy in four layers.

### 1. Message energy

Each message produces energy based primarily on message length.

Minimum recommended rule:

```text
message_energy = base_cost + (len(text) * per_character_cost)
```

Optional refinements:

- punctuation weight
- repeated-message penalty
- burst penalty for many short messages sent rapidly
- cross-channel routing penalty

### 2. Per-student cumulative energy

Each student maintains a cumulative energy score:

```text
student_energy[cms] += message_energy
```

This score represents how much channel stress that student is contributing over time.

### 3. Per-channel energy

Each channel maintains total channel energy:

```text
channel_energy[channel] = sum(student_energy of members currently on that channel)
```

This value is the main congestion signal.

### 4. Idle decay

Energy must decay when there is no activity.

Without decay, all channels eventually become permanently overloaded, which is not realistic and makes reallocation meaningless.

Recommended rule:

```text
new_energy = old_energy * decay_factor
```

Run decay:

- every few seconds in the AI observation loop, or
- whenever a channel snapshot is refreshed

Recommended starting values:

- global decay tick: every 5 seconds
- student decay factor: `0.90` to `0.97`

That means idle users gradually cool down instead of staying hot forever.

## Classification Model

Channel state should be based on total channel energy, not a single message.

Recommended statuses:

- `FREE`
- `BUSY`
- `CONGESTED`
- `JAMMED`

Example threshold logic:

```text
if total_energy < free_threshold:
    FREE
elif total_energy < busy_threshold:
    BUSY
elif total_energy < jammed_threshold:
    CONGESTED
else:
    JAMMED
```

Optional additional features:

- derived SNR
- modulation label
- rolling congestion confidence

But the key decision signal should remain total channel energy.

## Messaging Rules

### Required behavior

- A message can be sent to any valid student.
- Sender must be authenticated.
- Recipient must exist.
- Sender and recipient may be on:
  - same channel
  - different channels

### Delivery behavior

On send:

1. validate sender
2. validate recipient
3. compute message energy
4. increase sender energy
5. recompute sender channel total energy
6. apply decay update if needed
7. classify sender channel
8. if sender channel remains safe:
   - save message
   - deliver to recipient
   - confirm to sender
9. if sender channel is overloaded:
   - trigger decision engine
   - possibly reallocate students
   - then decide whether the message is delivered, delayed, or rejected

### Recommended first implementation rule

For the next implementation, use:

- cross-channel delivery is allowed
- overload is evaluated on the sender's current channel

This keeps the logic simpler and still matches the project goal.

## Reallocation Logic

### Why reallocate

Reallocation exists to keep messaging flow smooth when one channel becomes too stressed.

### When to reallocate

Reallocation should trigger when:

- a channel reaches `CONGESTED` or `JAMMED`, and
- the decision engine finds a safe destination

### Candidate selection

Candidates should be ranked using student energy, not join time.

Recommended selection inputs:

- highest current student energy on the source channel
- fairness rotation pointer
- optional recent-move cooldown

### Fairness rule

Do not always move the same heavy sender first.

Use a round-robin pointer per source channel:

```text
reallocation_pointer[channel] -> index
```

Flow:

1. rank members by descending energy
2. rotate the ranked list using the channel pointer
3. test candidates in rotated order
4. after a successful decision, advance the pointer

This gives fairness while still respecting energy.

### Destination selection

A destination channel is valid only if it can absorb the moved student's energy without becoming overloaded.

Projected check:

```text
projected_destination_energy =
    current_destination_energy + carried_student_energy
```

Accept destination only if projected status stays below overload.

### Carried energy after move

When a student moves, they should not arrive with full raw energy and should not reset to zero either.

Use decayed carryover:

```text
carried_energy = student_energy * relocation_decay_factor
```

Recommended starting value:

- `relocation_decay_factor = 0.5`

### Minimum-move rule

Do not move extra students unnecessarily.

After each move:

1. recompute source channel energy
2. reclassify source channel
3. stop immediately if source is safe again

## Round-Robin Usage in This Project

Round-robin should appear in two places.

### 1. Join distribution

New logins can still be assigned using round-robin or least-stress round-robin.

Better version:

- rotate candidate channels fairly
- among safe channels, prefer one with lower projected energy

### 2. Reallocation fairness

Within an overloaded channel:

- do not always select the top energy student first
- rotate the candidate order fairly

This is the more important round-robin usage for congestion control.

## Idle Decay Design

Idle decay is required by the project goal.

### Recommended implementation

Every observation tick:

1. visit all active students
2. reduce energy scores by decay factor
3. clamp tiny values to zero
4. recompute channel totals

Example:

```text
if energy < epsilon:
    energy = 0
else:
    energy = energy * 0.95
```

This gives:

- realistic cooldown
- less permanent congestion
- more meaningful reallocation behavior

## Required Backend Changes

### `main.py`

Change message validation and routing so that:

- same-channel and cross-channel messaging are both allowed
- sender channel energy is updated on every send
- messaging result clearly says:
  - accepted
  - delayed
  - rejected
  - reallocated

WebSocket and REST message payload should stay:

```json
{"to": "403897", "text": "Hello"}
```

### `signal_physics.py`

Must own:

- message energy calculation
- student cumulative energy store
- per-channel energy snapshot
- idle decay tick
- relocation carry-decay

Add or confirm these functions:

- `compute_message_energy(...)`
- `update_energy_score(...)`
- `get_energy_score(...)`
- `apply_idle_decay(...)`
- `get_channel_energy_snapshot(...)`
- `decay_energy_on_reallocation(...)`

### `classifier.py`

Must classify based on total channel energy.

Add or confirm:

- `classify_channel(channel_id)`
- `classify_channel_projected(channel_id, additional_energy)`

### `allocator.py`

Must become the decision engine.

Required behavior:

- detect overloaded source channel
- rank members by energy
- apply fairness rotation
- test projected destination safety
- move minimum students needed
- preserve admin-forced jam states if that feature remains

### `channels.py`

Keep it as the in-memory registry, but not the message router.

It should expose:

- member lookup
- channel metadata
- helper accessors for energy summaries if useful

### `database.py`

Messages should persist:

- sender CMS
- recipient CMS
- message type
- content
- timestamps

This is important because cross-channel DMs must still be reconstructable later.

### Frontend

The frontend must stop pretending every student is always directly reachable in the same way.

It should show:

- my current channel
- my current channel energy/status
- all students
- whether each target is:
  - same channel
  - cross channel
  - unavailable

When a cross-channel message is sent, UI should still allow it if the backend policy allows it.

## Recommended Processing Flow

### Send flow

1. sender submits DM
2. backend validates recipient
3. backend computes message energy
4. sender energy increases
5. idle decay is applied if due
6. sender channel total is recomputed
7. classifier evaluates sender channel
8. if safe:
   - deliver message
9. if overloaded:
   - allocator tests moves
   - moves minimum users needed
   - message is either:
     - delivered after stabilization
     - delayed
     - rejected if still unsafe

### AI loop

Every 5 seconds:

1. apply idle decay
2. observe all channels
3. classify all channels
4. for overloaded channels:
   - run fair reallocation
5. persist updated statuses

## Practical Next Implementation Order

### Phase 1

- allow cross-channel messaging
- keep energy tracked on sender channel
- expose clearer message-result errors in UI

### Phase 2

- add proper idle decay
- classify based on decayed cumulative totals

### Phase 3

- refine allocator with projected safety and fairness rotation

### Phase 4

- improve frontend to display same-channel versus cross-channel status clearly

## Final Design Decision Summary

The revised project direction is:

- cross-channel messaging is allowed
- energy is cumulative but not permanent
- energy increases with message activity
- energy decreases during idle time
- channel congestion is based on total channel energy
- reallocation moves students fairly using round-robin rotation over energy-ranked candidates
- destination channels must have spare energy before accepting a moved student
- the system should keep communication smooth rather than permanently blocking traffic
