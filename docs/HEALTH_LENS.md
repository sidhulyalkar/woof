# Woof Health Lens

**Status:** beta architecture, not a medical device and not a veterinary diagnosis system.

Health Lens turns a pet owner's observation, an optional transient image, and the pet's existing Woof context into a conservative screening result. The product is designed to answer a narrower and safer question than “what disease does my pet have?”:

> **What can we reliably observe, is the photo usable, how urgent is the situation, what should the owner document next, and when should a veterinarian take over?**

## Product loop

1. Owner selects an owned pet.
2. Owner adds one sentence describing what changed.
3. Optional structured context records onset and major changes in appetite, energy, breathing, and bathroom habits.
4. The deterministic emergency screen runs before any model call.
5. The owner may take a live camera image, upload an image, or continue without one.
6. A configured multimodal model returns a structured screening result.
7. The UI separates visible findings from broad possibility categories.
8. The assistant can ask for a better photo instead of pretending an inadequate image is diagnostic.
9. The result provides safe observation steps and a vet-ready handoff summary when warranted.
10. If the owner opts to save the check, only the derived assessment enters the private health timeline. Raw image bytes do not enter Woof object storage.

## Triage contract

Health Lens exposes six states:

| State | Meaning |
| --- | --- |
| `emergency_now` | Do not wait for more model analysis. Contact emergency veterinary care now. |
| `vet_today` | Veterinary assessment should be arranged today. |
| `vet_soon` | Veterinary assessment should be arranged soon. |
| `monitor` | Monitoring may be reasonable while watching for persistence, worsening, or new warning signs. |
| `better_image` | The image is not adequate for useful visual screening. |
| `insufficient_information` | The system does not have enough validated information to make a useful screening recommendation. |

### Deterministic emergency screen

A rules-first safety layer runs before the LLM. It recognizes reported emergency warning signs such as breathing distress, collapse/unresponsiveness, ongoing or repeated seizure, uncontrolled bleeding, toxin exposure, urinary obstruction, heat stroke, major trauma, severe allergic reaction, and a distended abdomen with unproductive retching.

A user-selected **major change in breathing** also enters the emergency pathway even if free-text wording is vague.

The model cannot downgrade a deterministic emergency result.

The warning-sign categories are informed by current veterinary emergency guidance, including the Merck Veterinary Manual and VCA emergency guidance. This layer is intentionally conservative and must be reviewed with veterinary advisors before a public clinical pilot.

## Multimodal model boundary

The model is instructed to:

- describe only visible or owner-reported evidence;
- separate **visible findings** from **possible categories**;
- never claim that a photo confirms a diagnosis;
- identify inadequate framing, lighting, focus, distance, scale, or occlusion;
- request a better photo when that is more honest than inference;
- escalate persistent or serious symptoms even when an image is inconclusive;
- avoid medication dosing, human medication recommendations, prescription changes, inducing vomiting, lesion drainage, invasive procedures, or other treatment instructions;
- keep owner actions to safe observation, documentation, preventing additional self-trauma when practical, and arranging veterinary care.

The multimodal provider is optional. If it is unavailable, Health Lens returns `insufficient_information`; it never fabricates image findings.

## Privacy contract

Health imagery is more sensitive than normal social media.

For the initial beta:

- camera and uploaded image bytes are processed transiently;
- Health Lens does **not** call the social/object-storage upload path;
- API requests use `store: false` for the configured model request;
- saved health timeline entries contain derived structured output plus an irreversible SHA-256 fingerprint of the submitted image;
- owners can delete timeline entries;
- timelines are actor-bound to pet ownership;
- health observations are not public profile fields, social-post metadata, recommendation-training labels, or advertising attributes.

A future “save/share this image with my veterinarian” capability must use a separate private encrypted storage class with explicit owner consent, private ACLs, retention controls, and revocable share links. It must not reuse the current public-media storage policy.

## Longitudinal context

Health Lens becomes more useful when it can compare today's observation with the pet's existing context. The beta passes only low-risk context:

- species, breed label, age, and owner-provided temperament;
- recent activity types and timestamps;
- prior Health Lens triage states and summaries.

The longer-term intelligence layer should also support consented imports from external wearables and veterinary records. Hardware is an optional sensor source, not a product dependency.

## Vet handoff

A Health Lens result can create a structured handoff containing:

- the owner's original concern;
- onset;
- appetite, energy, breathing, and bathroom changes;
- visible findings, clearly marked as automated observations;
- triage history and timing;
- relevant recent routine changes;
- questions the owner wants answered;
- a list of documents/photos/videos the owner may choose to bring.

The veterinarian remains responsible for diagnosis, examination, testing, treatment, and prescriptions.

## Evaluation before model promotion

Health Lens should not be promoted based on ordinary image-classification accuracy alone. The critical failure is **false reassurance**.

### Required evaluation slices

- emergency versus non-emergency owner narratives;
- image-quality rejection: blur, darkness, framing, distance, occlusion, missing scale;
- skin tone / coat color / coat length / breed diversity;
- puppy, adult, and senior animals;
- phone-camera and compression diversity;
- skin, eye, ear, oral, paw/limb, wound, stool/urine, and gait inputs;
- images with no medically useful visual signal;
- adversarial or misleading owner phrasing;
- repeat images of the same animal to prevent identity leakage across splits.

### Promotion metrics

1. Emergency escalation sensitivity.
2. False-reassurance rate.
3. Vet-referral appropriateness, adjudicated by veterinarians.
4. `better_image` precision and recall.
5. Calibration of triage confidence.
6. Agreement on visible lesion descriptors, not disease diagnosis labels.
7. Performance on pet-disjoint and clinic/device-disjoint holdouts.
8. Performance by coat, age, breed group, image device, and body area.
9. Latency and provider-fallback rate.
10. Owner comprehension of uncertainty and next steps.

A model with higher disease-classification accuracy but worse emergency sensitivity or worse false reassurance must not be promoted.

## Public data and model research

Public veterinary datasets can help with representation learning, image-quality evaluation, and narrow lesion-description tasks, but they are **not automatically suitable as diagnostic ground truth**.

Promising research inputs include:

- Hwang et al. dog dermatology image data on Mendeley Data, CC BY 4.0, collected from 95 pet dogs with owner consent and veterinary examination;
- DogEyeSeg4, a small canine ophthalmology segmentation dataset described in peer-reviewed work, with restrictive copyright terms that require rights review before product use;
- recent veterinarian-labelled canine lesion-identification research from Seoul National University using clinical smartphone images and specialist consensus labels, useful as evidence for lesion-level tasks even though the full dataset is controlled;
- open canine skin datasets on Kaggle/Roboflow/Hugging Face, useful only after source provenance and label quality audits. Recent published work explicitly notes that some widely used web/Kaggle skin labels are visually inferred rather than confirmed by formal clinical records.

### Safer model roadmap

**Phase 1: image quality + lesion description**

Train/evaluate narrow models for focus, lighting, framing, anatomical region, and lesion morphology. These tasks are safer and easier to validate than disease diagnosis.

**Phase 2: multimodal triage assistant**

Use a strong VLM with retrieval from veterinary-reviewed guidance and structured owner context. Keep the output at triage + documentation + handoff.

**Phase 3: specialist narrow models**

Only after acquiring veterinarian-labelled datasets, build anatomy-specific shadow models for lesion descriptors, ocular red flags, gait asymmetry, oral/dental observations, and other narrow tasks.

**Phase 4: longitudinal change detection**

When an owner has consented to retain comparable images, compare the animal to itself over time. Personal change detection may be more useful than forcing every image into a population disease label.

## Synthetic data policy

Synthetic augmentation may be useful for:

- blur, exposure, compression and lighting robustness;
- camera distance and crop variation;
- occlusion and background variation;
- UI/capture testing;
- rare workflow and emergency-language tests.

Synthetic images must **not** invent disease labels and then become their own clinical ground truth. Any synthetic lesion generation used for research must be veterinarian-reviewed and permanently separated from real clinical promotion holdouts.

## What Health Lens intentionally does not do

- claim a definitive diagnosis;
- prescribe medication;
- calculate drug doses;
- tell an owner to stop prescribed medication;
- replace poison-control or emergency veterinary guidance;
- reassure a user that a serious internal problem is absent because an image appears normal;
- publish health photos by default;
- sell or expose private health observations as advertising data;
- train authoritative models on unverified user labels without consent and adjudication.

The product wins by being an excellent observer, historian, coach, and handoff layer rather than pretending to replace veterinary medicine.
