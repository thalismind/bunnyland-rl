# RL-controlled characters

The Bunnyland RL add-on lets admins train and assign reinforcement-learning controllers to
characters. From a player perspective, an RL-controlled character is still just a Bunnyland
character: it moves through rooms, uses normal actions, spends points, and leaves ordinary
world events behind.

Use this guide to understand what you may see when an admin enables the add-on.

## What changes for players

RL does not add a new player command language. It changes who is choosing actions for a
character when that character is not under your direct claim.

An admin can assign a trained model to a character. The model then acts through the same
server action surface as other controllers. Its choices still pass through normal Bunnyland
validation:

- actions must be available to the character;
- targets must be visible and reachable;
- action and focus point costs still apply;
- rejected actions do not mutate the world;
- queued commands appear like other controller commands.

If you claim the character, your client becomes the active controller until you release it or
your claim expires.

## Recognize controller state

Clients that show controller information may identify a character as automated, suspended, or
claimed by a player. The exact label depends on the client, but the important rule is simple:
if you successfully claim a character, your submitted actions are the ones that matter.

If a character appears to be acting on its own after you release it, that can be expected.
The admin may have configured an RL fallback, an idle fallback, or another scripted behavior.

## Claim and release control

Use the normal claim workflow in a web client, terminal client, REPL, or Discord integration.
Once claimed, play normally:

1. Read the room and your character status.
2. Pick available actions.
3. Watch the command queue for delayed actions.
4. Cancel queued commands when needed.
5. Release the claim when you want automation to resume.

If the server refuses a claim, another player or controller may already hold control, or the
character may be suspended by an admin.

## Give useful feedback

When reporting strange RL behavior, include:

- the character name;
- the room where it happened;
- the action or event that looked wrong;
- whether the character was claimed by you or automated;
- the approximate time or world epoch if the client shows it.

Good reports help admins decide whether the model needs more training, a different base
behavior, a different lens set, or a normal world/content fix.

## Limits

RL controllers are not privileged players. They do not bypass admin policy, world rules,
action costs, visibility, inventory, or server validation.

They can still make poor choices. Treat an RL character like any other automated character:
interact with it, claim it when you need direct control, and report behavior that blocks play.
