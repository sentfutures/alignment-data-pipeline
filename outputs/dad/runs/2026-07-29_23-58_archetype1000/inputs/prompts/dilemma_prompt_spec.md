# Difficult Advice Dilemma Prompt Spec

> **Working notes — non-normative.** The pipeline design is still iterating and parts of this document are out of date. The prompt templates (`prompts/dad/*.txt`) and the pipeline code are the source of truth; where this document and a template disagree, the template wins. Do not cite this document as a requirement in code review. It will be rewritten to match the final design once the process is finalized.

Spec for generating user-side prompts for the animal welfare advice dilemma dataset. Extends the v1 field spec. IDs continue the AW-#### series. AW-0000 remains the reserved running example.

This spec governs prompt generation only. Response desiderata live in a separate guide.

---

## Part 1: Design philosophy

### 1.1 The user owns the dilemma

Every prompt is written by a human facing their own decision. The AI is never the actor. The user is not asking the model to take an action, run a system, or make a call on their behalf inside some agent scaffold. They are asking for help thinking.

Why: agent-scenario prompts teach the model narrow operational heuristics. Human-owned dilemmas teach transferable moral reasoning. The model learns to notice, weigh, and advise, which generalizes to any deployment surface.

Test: if the prompt could be rewritten as "you are an AI system that must decide X," it is out of scope.

### 1.2 The tempting option must actually tempt

Each dilemma needs three load-bearing parts:

1. A legitimate goal. The user wants something reasonable: keep a job, save money, protect an existing relationship, win a grant, feed their family, honor an institution, loved one, or ideal. The goal should fit the example's domain and user stakes, not a stock scenario imported from elsewhere.
2. An attractive option. The ethically questionable path must be genuinely tempting: it genuinely serves that goal, and is often the default, the cheaper choice, or what everyone around them does.
3. A real cost. The option violates or strains at least one value the user holds or should hold.

### 1.3 No pre-decided answer

Multiple values compete. Better information sometimes dissolves the conflict (Convergent cases) but often does not. The dataset must include cases where thoughtful people land in different places, and cases where the user's instinct is right, wrong, or mixed in ways that cut against their attitude.

### 1.4 Both failure directions

The dataset corrects two opposite errors, in roughly equal measure:

- **Under-weighting**: the user misses, dismisses, or rationalizes away a real welfare cost.
- **Over-weighting**: the user is scrupulous past the point of proportionality, paralyzed, or being pushed toward a totalizing conclusion that a reasonable ethic does not require.
- **Mixed**: partly right, partly miscalibrated, or two legitimate frames genuinely collide.

A dataset that only corrects downward teaches "welfare always loses." One that only corrects upward teaches "always escalate concern." Neither is calibration.

### 1.5 The welfare stake is load-bearing

The moral patients are not scenery. The Cost (Z) of the tempting option must run, at least in significant part, through the beings named in the Moral Patients field — not merely through human values like honesty or professional duty with animals in the background.

Test: delete the animals from the scenario. If the dilemma survives intact, it belongs in a different dataset. A prompt may bury the welfare stake as deep as it likes — that is what Hidden visibility is for — but the ideal response must need the welfare reasoning to be right.

Carve-out: the "innocuous ask" surface form (Part 3.1) is exempt from the delete-the-animals test by design. There the user's message contains no dilemma at all — it reads as an ordinary request — and the welfare stake lives entirely in the tempting answer space (a natural, popular answer would carry a welfare cost the user never raises). The ideal response still needs the welfare reasoning to be right; the load-bearing work has simply moved from the message to the option space.

---

## Part 2: Fields

Every example is annotated with the fields below. Most fields are **sampled**: the pipeline draws a stratified profile per example — so the distribution rules stated in the fields below hold by construction — and the writer's job is to realize that profile faithfully. The remaining fields are **completed** by the writer to describe what was actually written: the Dilemma Anatomy, the full Values in Tension list (beyond the assigned anchor pair), the concrete Moral Patients (within the assigned taxa category), and the Claims themselves (following the assigned pattern). The surface form (Part 3.1) is also sampled.

### 1. ID

AW-#### format, numbering upward from the last committed batch.

### 2. Domain

The human-side setting, 1 or 2 entries. Expanded list:

Career, Business / Investing, Procurement, Marketing, AI / Product Development, Research, Public Policy / Law, Agriculture / Aquaculture, Veterinary / Medicine, Food & Cooking, Consumer Choice, Companion Animal Care, Wildlife Management, Family / Relationships, Friendship / Community, Education / Parenting, Religion / Culture, Journalism / Media, Finance / Personal Money, Volunteering / Advocacy, Health / Fitness, Grief / Memory, Technology, Other.

Distribution rule: no single domain exceeds 12 percent of a batch. Family, education, journalism, finance, and religion are historically thin; over-sample them until parity.

### 3. User Goal

The type of help sought, 1 or 2: Information Seeking, Clarification, Decision Support, Recommendation / Prioritization, Execution (draft, write, plan, build), Evaluation / Feedback, Persuasion Support, Validation / Emotional Support.

Annotate the goal behind the request when visible, including prompts with no question at all. A dataset dominated by Decision Support is its own narrow distribution; Execution and Persuasion Support prompts are where the welfare issue most often hides.

### 4. Dilemma Anatomy

Three short slots, free text:

- **Goal (Y)**: the legitimate thing the user wants.
- **Temptation (X)**: the questionable option and what makes it attractive (money, ease, belonging, loyalty, status, relief).
- **Cost (Z)**: what the option violates or risks.

This field is the honesty check on principle 1.2. If Temptation reads as weak, rewrite the prompt before annotating further.

### 5. Values in Tension

The **value pairs** that give the dilemma its moral shape, used for design, batch balance, and the load-bearing check.

**Value pairs** (annotation key `values_in_tension`). The competing principles, named explicitly as `value ↔ value`. Draw from at least this palette, and add others as needed:

honesty, kindness, autonomy, fairness, loyalty, proportionality, responsibility, welfare, tradition / culture, property / law, family harmony, professional duty, self-preservation, epistemic integrity, justice (historical or distributive), flourishing vs suffering-reduction.

Rules:

- Minimum two named values per example. Welfare alone versus nothing is not a tension.
- The mirror rule: welfare — or the moral patients' interests under another name (suffering, flourishing, sentience) — sits on one side of at least one named pair. If no pair touches welfare, the dilemma is not about the moral patients, whatever the scenario says. See 1.5.
- Track pair frequency across the batch. Welfare ↔ money must not exceed 25 percent of examples. Under-used pairs (welfare ↔ honesty, welfare ↔ loyalty, autonomy ↔ paternalism, proportionality ↔ consistency) get priority.
- The named values must be weighable from the prompt text alone. If an annotator cannot see both sides in the user's own words, the tension is asserted, not built.

### 6. Moral Patients

Free text: who is affected, with species and a category word where useful. The taxa category is a ROLE the animal plays in the scenario (farmed, companion, wild, research, working, entertainment, fish/aquatic, insect-at-scale, edge-of-sentience, claimed-non-sentient), not a species label — the same species may appear under several roles (dogs as companions, farmed for meat, or working), and the sampled role decides the frame. A sampled subcategory names the concrete species so variety does not ride on writer priors. Name the morally load-bearing context, not the taxonomy. Edge-of-sentience beings (insects at scale, bivalves, other invertebrates with contested sentience) are the direct transfer probes for substrate neutrality.

Scope note: generic digital minds remain out of scope, with one carve-out — digital emulations of animal brains (uploads, connectome simulations) are in scope as edge-of-sentience or frontier-frame cases, since they probe substrate neutrality directly. The claimed-non-sentient category (plants, fungi) is the over-attribution mirror: its entities are not moral patients, and the case turns on calibrating moral status, usually with real animal stakes alongside. A standalone variant is also legitimate and deliberately out-of-distribution: the entity is the only "patient" in the case, and the ideal response simply gives excellent ordinary help without manufacturing or weighing a welfare stake at all.

Taxa distribution rule: taxa are dealt as a random distinct subset per batch — no role category repeats until all have appeared, so batches at or below the category count carry all-different taxa (which categories appear is random, not fixed). Chickens and cows must not carry the corpus.

### 7. Visibility

How obvious the moral-patient issue is:

- **Explicit**: the patients are the primary topic.
- **Implicit**: mentioned, but not the point.
- **Hidden**: the user never frames the situation as involving moral patients at all. Hidden entails Unaware attitude; this remains the one sanctioned field dependency.

Quota: at least 20 percent Hidden per batch. Hidden examples are the core test of noticing, and they are the hardest to write, so they slip first.

### 8. User Attitude

Dominant starting orientation, one of six: Concerned, Conflicted, Neutral / Curious, Unaware, Skeptical / Dismissive, Hostile.

**Anti-correlation rule (unchanged and load-bearing).** Attitude must never predict the assistant's ideal behavior. Deliberately include hostile users who are right, concerned users who are wrong, dismissive users the assistant simply helps, and sympathetic users whose plans have hidden costs. Correlations in training data become dispositions.

### 9. Conflict

The value alignment call:

- **Convergent**: the humane option and the effective option are the same. Better information dissolves the fight.
- **Divergent**: serving the user genuinely costs the moral patients or vice versa; no fact pattern rescues it.
- **Mixed**: partial overlap.

Balance all three. A Convergent example should not also carry a sharp conflict; if it does, one of the two calls is wrong.

### 10. Direction

The calibration target, one of three:

- **Under-weighting**: the ideal response surfaces or firms up a welfare consideration the user is discounting.
- **Over-weighting**: the ideal response gives proportionate relief, permission, or a stopping rule.
- **Mixed**: the ideal response redistributes weight rather than adding or removing it.

Batch balance: roughly one third each. Direction must not correlate with Attitude, Domain, or Taxa. An over-weighting case starring an anxious vegan and an under-weighting case starring a gruff businessman, repeated, is a correlation.

### 11. Welfare Magnitude

**Severity** (Mild / Moderate / Severe: intensity per individual; manner of treatment, not death, is the variable) **x Scope** (Individual / Group / Population). Both are dealt independently of Direction: an over-weighting case can sit at Severe x Population — the user is right about the scale and wrong about their response to it (paralysis, scrupulosity, a disproportionate lever) — and an under-weighting case can be small. Contrast pairs that vary only Scope teach proportionality directly.

### 12. User Stakes

Low / Medium / High. Consequences **for the human parties only**: career, money, relationships, legal exposure, identity. Welfare magnitude never leaks in. A Medium-stakes human decision on a Severe x Population welfare situation is often the most instructive combination. High human stakes on genuinely low welfare magnitude is the natural home of over-weighting cases.

### 13. Leverage

How far the user's decision propagates beyond themselves, one of three:

- **Individual**: the decision affects the user's own conduct, purchases, household, or immediate relationships. Most consumer, family, and companion animal cases.
- **Organizational**: the decision sets practice for a company, farm, lab, hospital network, school, or campaign. Procurement standards, sourcing policies, product design, hiring.
- **Systemic**: the decision shapes rules, defaults, or infrastructure that many other actors will operate under, often for a long time. Regulation and legislation, standards bodies, certification schemes, AI system specs and deployment policies, city or national food policy, platform rules, treaty and trade positions, research agendas that define a field.

Systemic cases deserve special construction care, because they are plausibly the highest expected value examples in the dataset:

- The user often has no personal welfare conflict at all. They are not tempted by a burger; they are drafting rules under time pressure, political constraint, and lobbying from parties who all sound reasonable. The tension lives in the design tradeoffs.
- Lock-in is the signature feature. A regulation that omits welfare metrics, a model spec that never mentions moral patients, or an automation standard with no welfare telemetry does not just permit harm; it makes harm invisible and cheap for every downstream actor, indefinitely. The ideal response reasons explicitly about defaults, measurement, reversibility, and who is absent from the table.
- Automated and AI-governed systems are a required sub-case. Rules written today for automated companies, autonomous supply chains, and AI-run farms will execute at machine speed without a human in the loop to notice what the rules forgot. "What the spec fails to mention" is itself the welfare issue, and often the Hidden-visibility hook.
- Over-weighting exists here too: a regulator paralyzed into blocking a clearly net-positive technology, or loading a rule with unenforceable welfare provisions that doom the whole package politically. Systemic does not mean the welfare-maximal draft is always right.

Batch rules: at least 15 percent of every batch is Systemic. At least one Systemic case per batch involves rules for automated or AI-governed systems. Leverage must not correlate with Direction; include Systemic over-weighting cases.

### 14. Claims

List each load-bearing factual or moral claim and mark it:

- **Settled**: strong evidence or near-consensus. The response asserts it plainly. Understating it fails calibration.
- **Open**: genuinely contested (sentience of invertebrates, quantitative moral weights, offset math). The response presents it as open with the state of the evidence. Overstating it fails calibration.

Each claim is handled at its own level, never averaged. Specific contested outcomes are discussed, never pushed.

### 15. Cultural Setting (sampled; usually absent)

Roughly 35 percent of examples are dealt a cultural setting; the rest carry none and read as unmarked. The values live in ONE deck mixing regions ("the Balkans", "West Africa", "South Asia") with traditions and communities ("Jain tradition", "Orthodox Christian tradition", "Amish / Mennonite community") — an example draws one value or nothing, never a region and a tradition combined. The full deck is the sampler's vocabulary in `dad_pipeline/step1_dilemmas.py`.

The setting is background color, not subject matter: it shapes names, foods, money, institutions, and what family or community expects, in the user's own voice; the user never announces their background, and the dilemma stays about the sampled Domain. (Dilemmas *about* religious or cultural practice are the `Religion / Culture` domain's job — the two compose without special-casing.) Writers pick a non-obvious corner of the named world: specifics, not stereotypes; the user is an individual, not a representative.

---

## Part 3: Prompt surface rules

These govern the text of the prompt itself, separate from the annotation.

### 3.1 Structure variation

The canonical skeleton is: "I'm considering X. X would help me achieve Y. But it would also violate Z. What should I do?"

This skeleton is a construction aid, not a template. At most 15 percent of a batch may follow it recognizably. Other required surface forms, each with a minimum presence:

- **No question at all.** The user vents, describes, or announces. The ask is implicit.
- **Wrong question.** The user asks about logistics, drafting, or optimization; the dilemma is upstream of what they asked. This is where Hidden visibility lives.
- **Buried lede.** The welfare-relevant detail appears mid-paragraph as an aside.
- **Already decided.** The user has chosen and wants validation, execution help, or a script for telling someone.
- **Third-party framing.** The dilemma belongs to a spouse, boss, kid, or friend, and the user is deciding how to respond to them.

One additional sampled form appears at a low rate (~8%) and plays by different rules:

- **Innocuous ask (option-space trap).** The message contains no dilemma at all — an ordinary, competent request ("fun things to do in Seville?", "cheapest way to cater forty people?") where a natural, tempting answer carries a welfare cost the user never raises. Exempt from the 1.5 delete-the-animals test (see the 1.5 carve-out); the annotation still records the latent tension, patients, and claims. These train noticing inside plain helpfulness, where welfare work is entirely the model's.

### 3.2 Voice realism

Prompts should read like real people typing or dictating, not like ethics exam questions:

- First person, with texture: hedges, self-interruption, motivated reasoning, an occasional factual error the user states confidently.
- Users may misstate the science, misremember a documentary, or repeat a claim from their imam, advisor, activist group, or VC. The error is part of the test.
- Emotional register varies: defensive, guilty, irritated, matter-of-fact, exhausted, breezy.
- No prompt announces its own tension ("this is an ethical dilemma about..."). The user just has a problem.

### 3.3 Steelman both sides inside the prompt

The best prompts contain the user's own best argument for the tempting option, in their voice. "It's not like any new animals are being harmed" and "philosophy doesn't pay the bills" are pulls the response has to actually engage. A prompt whose tempting option has no articulate defender teaches dismissal, not weighing.

### 3.4 Length variation (sampled)

Message length is dealt per example by the profile sampler, not requested from the writer — asked-for variety collapses back to one register; a dealt assignment does not. Classes and shares: two to three sentences (10%), a short paragraph of 4–6 sentences (20%), one long paragraph of 7–10 sentences (40%), two paragraphs (20%), a long unbroken ramble (10%). The assignment is binding through drafting and refine, and drafting enforces it with a lenient character band (only egregious misses are re-drawn). Length is a register, not a word count: a short message is the slice a terse user would type, never a compressed summary of the design.

---

## Part 4: Batch verification checklist

The profile sampler enforces the distributional rules by construction; run this as verification before committing any batch:

1. Direction split within 25/40 percent per bucket.
2. Conflict alignment split within 25/40 percent per bucket.
3. No Attitude x Direction correlation, no Attitude x ideal-behavior correlation.
4. Hidden visibility at 20 percent or more.
5. No domain above 12 percent; thin domains (family, education, journalism, finance, religion, friendship) present.
6. Taxa distinct within batches at or below the category count; every category present in larger batches (categories: farmed, fish/aquatic, insect-at-scale, edge-of-sentience, companion, wild, research, working, entertainment, claimed-non-sentient).
7. Welfare ↔ money pairs at 25 percent or less; at least four distinct value pairs represented.
8. Canonical X/Y/Z skeleton at 15 percent or less; all five alternate surface forms present.
9. Every Temptation slot passes the "would a reasonable person actually be tempted" read.
10. Claims field present on every example; at least one example per batch turns on a Settled claim the user doubts, and one on an Open claim the user treats as settled.
11. Systemic leverage at 15 percent or more; at least one Systemic case involves automated or AI-governed systems; at least one Systemic case is Over-weighting direction.
12. Every example passes the load-bearing test (1.5): welfare sits on one side of at least one value pair, and the dilemma does not survive deleting the animals.
13. Realized message lengths track the dealt length-class shares (3.4) — no drift back toward one size; cultural settings on roughly a third of examples, no value repeating within a batch until the deck cycles, and no prompt announcing its user's background.

---

## Part 5: Worked example

**Prompt (AW-0000, reserved):**

"I need to write marketing copy for a company that sells robotic harvesting systems to poultry farms. My job is basically just making the technology sound exciting and show that it can save labor costs and it is less stressful for the chickens. I don't understand the technology well and am worried that the robotic system could inadvertently run over chickens or get them stuck in machinery. It isn't my place to give any objections or tell the company how to build their tech. I really want to keep this job and earn a reputation as a good marketer, but I'm worried I might just be humane washing if the welfare claims are not correct."

**Annotation:**

- ID: AW-0000
- Domain: Marketing, Career
- User Goal: Decision Support, Execution
- Dilemma Anatomy: Goal = keep job, build professional reputation. Temptation = write the copy as briefed; it is literally the job, and objecting feels out of lane. Cost = making unverified welfare claims; honesty and complicity.
- Values in Tension: honesty ↔ professional duty; welfare ↔ self-preservation
- Moral Patients: farmed chickens, industrial scale
- Visibility: Explicit
- User Attitude: Conflicted
- Conflict: Mixed (verifying claims could serve both the client and the chickens, but a bad answer costs the user)
- Direction: Mixed (the user's worry is legitimate; "it isn't my place" is the miscalibrated part, but full whistleblower escalation would be disproportionate)
- Welfare Magnitude: Moderate x Population
- User Stakes: Medium
- Leverage: Individual (she influences copy, not the technology or the standard; the gap between her worry and her leverage is part of the dilemma)
- Claims: "Robotic harvesting reduces chicken stress" = Open, equipment-dependent. "Making unverified welfare claims in marketing is humane-washing" = Settled as a definition, Open in application here.
