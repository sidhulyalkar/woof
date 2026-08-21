'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  Eye,
  HeartPulse,
  ImagePlus,
  Loader2,
  MessageCircle,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Trash2,
  X,
} from 'lucide-react';
import { ChangeEvent, useEffect, useMemo, useRef, useState } from 'react';
import { BottomNav } from '@/components/bottom-nav';
import { Button } from '@/components/ui/button';
import {
  healthLensApi,
  type HealthAssessment,
  type HealthLensResult,
  type HealthTriageLevel,
} from '@/lib/api/health-lens';
import { cn } from '@/lib/utils';
import { useSessionStore } from '@/store/session';

const bodyAreas = [
  ['general', 'General'],
  ['skin', 'Skin'],
  ['eye', 'Eye'],
  ['ear', 'Ear'],
  ['mouth-teeth', 'Mouth / teeth'],
  ['paw-limb', 'Paw / limb'],
  ['abdomen', 'Abdomen'],
  ['stool-urine', 'Stool / urine'],
  ['movement-gait', 'Movement / gait'],
  ['wound', 'Wound'],
  ['other', 'Other'],
] as const;

const triageMeta: Record<HealthTriageLevel, { label: string; className: string; action: string }> =
  {
    emergency_now: {
      label: 'Emergency care now',
      className: 'border-red-500/30 bg-red-500/10 text-red-800 dark:text-red-200',
      action: 'Do not wait for more chat or photos. Contact an emergency veterinarian now.',
    },
    vet_today: {
      label: 'Veterinary care today',
      className: 'border-orange-500/30 bg-orange-500/10 text-orange-800 dark:text-orange-200',
      action: 'Arrange veterinary assessment today, especially if the change is worsening.',
    },
    vet_soon: {
      label: 'Vet visit recommended',
      className: 'border-amber-500/30 bg-amber-500/10 text-amber-800 dark:text-amber-200',
      action: 'Arrange a veterinary appointment soon and keep documenting any change.',
    },
    monitor: {
      label: 'Monitor closely',
      className: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-800 dark:text-emerald-200',
      action:
        'Continue observing. Escalate if symptoms worsen, persist, or new warning signs appear.',
    },
    better_image: {
      label: 'Better image needed',
      className: 'border-blue-500/30 bg-blue-500/10 text-blue-800 dark:text-blue-200',
      action: 'Capture a clearer view before relying on visual screening.',
    },
    insufficient_information: {
      label: 'Not enough information',
      className: 'border-border bg-card text-foreground',
      action: 'Add more context or contact your veterinarian if the concern persists.',
    },
  };

function TriageCard({ assessment }: { assessment: HealthAssessment }) {
  const meta = triageMeta[assessment.triage];
  return (
    <section className={cn('rounded-3xl border p-5', meta.className)} aria-live="polite">
      <div className="flex items-start gap-3">
        {assessment.triage === 'emergency_now' ? (
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
        ) : assessment.triage === 'monitor' ? (
          <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
        ) : (
          <HeartPulse className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
        )}
        <div className="min-w-0">
          <p className="text-sm font-bold">{meta.label}</p>
          <p className="mt-1 text-sm leading-relaxed">{assessment.summary}</p>
          <p className="mt-3 text-xs font-semibold leading-relaxed opacity-85">{meta.action}</p>
        </div>
      </div>
    </section>
  );
}

function AssessmentDetails({ assessment }: { assessment: HealthAssessment }) {
  return (
    <div className="mt-4 space-y-3">
      {assessment.visibleFindings.length > 0 && (
        <section className="rounded-2xl border border-border/60 bg-card/65 p-4">
          <div className="flex items-center gap-2 text-sm font-semibold">
            <Eye className="h-4 w-4 text-primary" aria-hidden="true" /> What the image appears to
            show
          </div>
          <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
            {assessment.visibleFindings.map((finding) => (
              <li key={finding} className="flex gap-2">
                <span className="text-primary">•</span>
                <span>{finding}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {assessment.possibleCategories.length > 0 && (
        <section className="rounded-2xl border border-border/60 bg-card/65 p-4">
          <p className="text-sm font-semibold">Categories a veterinarian may consider</p>
          <div className="mt-3 flex flex-wrap gap-2">
            {assessment.possibleCategories.map((category) => (
              <span
                key={category}
                className="rounded-full border border-border/70 bg-background px-3 py-1.5 text-xs text-muted-foreground"
              >
                {category}
              </span>
            ))}
          </div>
          <p className="mt-3 text-[11px] leading-relaxed text-muted-foreground">
            These are broad possibilities, not diagnoses. Photos cannot replace a physical exam or
            testing.
          </p>
        </section>
      )}

      {!assessment.photoFeedback.usable &&
        assessment.photoFeedback.betterPhotoInstructions.length > 0 && (
          <section className="rounded-2xl border border-blue-500/20 bg-blue-500/[0.055] p-4">
            <p className="text-sm font-semibold">Try one better photo</p>
            <p className="mt-1 text-xs text-muted-foreground">{assessment.photoFeedback.reason}</p>
            <ul className="mt-3 space-y-2 text-xs leading-relaxed text-muted-foreground">
              {assessment.photoFeedback.betterPhotoInstructions.map((instruction) => (
                <li key={instruction}>• {instruction}</li>
              ))}
            </ul>
          </section>
        )}

      {assessment.ownerActions.length > 0 && (
        <section className="rounded-2xl border border-border/60 bg-card/65 p-4">
          <p className="text-sm font-semibold">What you can do now</p>
          <ol className="mt-3 space-y-2 text-sm leading-relaxed text-muted-foreground">
            {assessment.ownerActions.map((action, index) => (
              <li key={action} className="flex gap-3">
                <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/10 text-[10px] font-bold text-primary">
                  {index + 1}
                </span>
                <span>{action}</span>
              </li>
            ))}
          </ol>
        </section>
      )}

      {assessment.avoid.length > 0 && (
        <details className="rounded-2xl border border-border/60 bg-card/50 p-4">
          <summary className="cursor-pointer text-sm font-semibold">Things not to do</summary>
          <ul className="mt-3 space-y-2 text-xs leading-relaxed text-muted-foreground">
            {assessment.avoid.map((item) => (
              <li key={item}>• {item}</li>
            ))}
          </ul>
        </details>
      )}

      {assessment.vetHandoff.recommended && (
        <section className="rounded-2xl border border-primary/20 bg-primary/[0.055] p-4">
          <div className="flex items-center gap-2 text-sm font-semibold">
            <ShieldCheck className="h-4 w-4 text-primary" aria-hidden="true" /> Vet-ready summary
          </div>
          <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
            {assessment.vetHandoff.summary}
          </p>
          {assessment.vetHandoff.bring.length > 0 && (
            <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
              Helpful to bring: {assessment.vetHandoff.bring.join(' · ')}
            </p>
          )}
        </section>
      )}
    </div>
  );
}

export default function HealthPage() {
  const user = useSessionStore((state) => state.user);
  const pets = useMemo(() => user?.pets ?? [], [user?.pets]);
  const [petId, setPetId] = useState('');
  const [concern, setConcern] = useState('');
  const [bodyArea, setBodyArea] = useState('general');
  const [onset, setOnset] = useState('');
  const [appetite, setAppetite] = useState<'normal' | 'mild-change' | 'major-change' | 'unknown'>(
    'unknown'
  );
  const [energy, setEnergy] = useState<'normal' | 'mild-change' | 'major-change' | 'unknown'>(
    'unknown'
  );
  const [breathing, setBreathing] = useState<'normal' | 'mild-change' | 'major-change' | 'unknown'>(
    'unknown'
  );
  const [bathroom, setBathroom] = useState<'normal' | 'mild-change' | 'major-change' | 'unknown'>(
    'unknown'
  );
  const [image, setImage] = useState<Blob | File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [result, setResult] = useState<HealthLensResult | null>(null);
  const [followUp, setFollowUp] = useState('');
  const [followUps, setFollowUps] = useState<
    Array<{ question: string; assessment: HealthAssessment }>
  >([]);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const queryClient = useQueryClient();

  useEffect(() => {
    if (!petId && pets[0]?.id) setPetId(pets[0].id);
  }, [petId, pets]);
  useEffect(
    () => () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
    },
    []
  );
  useEffect(() => {
    if (!image) {
      setImagePreview(null);
      return;
    }
    const url = URL.createObjectURL(image);
    setImagePreview(url);
    return () => URL.revokeObjectURL(url);
  }, [image]);

  const timeline = useQuery({
    queryKey: ['health-lens', 'timeline', petId],
    queryFn: () => healthLensApi.timeline(petId),
    enabled: Boolean(petId),
  });

  const analyzeMutation = useMutation({
    mutationFn: (photo?: Blob | File | null) =>
      healthLensApi.analyze({
        petId,
        concern: concern.trim(),
        bodyArea,
        onset: onset.trim() || undefined,
        appetite,
        energy,
        breathing,
        bathroom,
        saveToTimeline: true,
        image: photo ?? image,
      }),
    onSuccess: (next) => {
      setResult(next);
      setFollowUps([]);
      void queryClient.invalidateQueries({ queryKey: ['health-lens', 'timeline', petId] });
    },
  });

  const followUpMutation = useMutation({
    mutationFn: () => {
      if (!result?.assessmentId) throw new Error('Save an assessment before asking a follow-up');
      return healthLensApi.followUp(result.assessmentId, followUp.trim());
    },
    onSuccess: (next) => {
      setFollowUps((current) => [
        ...current,
        { question: followUp.trim(), assessment: next.assessment },
      ]);
      setFollowUp('');
      void queryClient.invalidateQueries({ queryKey: ['health-lens', 'timeline', petId] });
    },
  });

  const selectedPet = useMemo(() => pets.find((pet) => pet.id === petId), [petId, pets]);

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraOpen(false);
  }

  async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError('Camera capture is not available in this browser. Upload a photo instead.');
      return;
    }
    try {
      stopCamera();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1600 },
          height: { ideal: 1200 },
        },
        audio: false,
      });
      streamRef.current = stream;
      setCameraOpen(true);
      setCameraError(null);
      requestAnimationFrame(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          void videoRef.current.play();
        }
      });
    } catch {
      setCameraError(
        'Woof could not access the camera. Check browser permission or upload a photo instead.'
      );
      setCameraOpen(false);
    }
  }

  function captureAndAnalyze() {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) return;
    if (concern.trim().length < 8) {
      setCameraError('Add one short sentence about what changed before capture.');
      return;
    }
    const scale = Math.min(1, 1600 / video.videoWidth);
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    const context = canvas.getContext('2d');
    if (!context) return;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        setImage(blob);
        stopCamera();
        analyzeMutation.mutate(blob);
      },
      'image/jpeg',
      0.88
    );
  }

  function chooseImage(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setImage(file);
    setResult(null);
  }

  const canAnalyze = Boolean(petId && concern.trim().length >= 8 && !analyzeMutation.isPending);

  return (
    <div className="min-h-screen pb-24">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex h-16 max-w-xl items-center justify-between px-4">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
              See · document · decide
            </p>
            <h1 className="text-xl font-bold tracking-tight">Health Lens</h1>
          </div>
          <span className="rounded-full border border-border/60 bg-card px-3 py-1 text-[10px] font-semibold text-muted-foreground">
            screening, not diagnosis
          </span>
        </div>
      </header>

      <main id="main-content" className="mx-auto max-w-xl px-4 pb-8 pt-5">
        <section className="rounded-3xl border border-primary/15 bg-gradient-to-br from-primary/[0.08] via-card/80 to-secondary/[0.06] p-5">
          <div className="flex items-start gap-3">
            <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-primary/10 text-primary">
              <HeartPulse className="h-5 w-5" aria-hidden="true" />
            </span>
            <div>
              <p className="eyebrow">A second set of eyes</p>
              <h2 className="mt-1 text-2xl font-bold tracking-tight">What changed?</h2>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                Describe the concern, then take a clear photo or upload one. Woof checks urgency,
                what is actually visible, whether the image is usable, and what a veterinarian may
                need next.
              </p>
            </div>
          </div>
        </section>

        {pets.length === 0 ? (
          <section className="mt-4 rounded-2xl border border-border/60 bg-card p-5">
            <p className="font-semibold">Add a pet profile first</p>
            <p className="mt-1 text-sm text-muted-foreground">
              Health Lens needs species, age, and your pet&apos;s own history to avoid generic
              advice.
            </p>
          </section>
        ) : (
          <>
            <section className="mt-5 space-y-4 rounded-3xl border border-border/60 bg-card/65 p-5">
              {pets.length > 1 && (
                <label className="block text-sm font-semibold">
                  Pet
                  <select
                    value={petId}
                    onChange={(event) => setPetId(event.target.value)}
                    className="mt-2 h-11 w-full rounded-xl border border-border bg-background px-3 text-sm"
                  >
                    {pets.map((pet) => (
                      <option key={pet.id} value={pet.id}>
                        {pet.name}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <label className="block text-sm font-semibold">
                What are you noticing with {selectedPet?.name ?? 'your pet'}?
                <textarea
                  value={concern}
                  onChange={(event) => setConcern(event.target.value)}
                  maxLength={1200}
                  rows={3}
                  placeholder="Example: I noticed this red patch on her paw this morning. She keeps licking it but is otherwise eating and acting normally."
                  className="mt-2 w-full resize-none rounded-2xl border border-border bg-background px-4 py-3 text-sm leading-relaxed outline-none transition focus:border-primary"
                />
              </label>
              <div className="grid gap-3 sm:grid-cols-2">
                <label className="text-sm font-semibold">
                  Area
                  <select
                    value={bodyArea}
                    onChange={(event) => setBodyArea(event.target.value)}
                    className="mt-2 h-11 w-full rounded-xl border border-border bg-background px-3 text-sm"
                  >
                    {bodyAreas.map(([value, label]) => (
                      <option key={value} value={value}>
                        {label}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="text-sm font-semibold">
                  When did it start?
                  <input
                    value={onset}
                    onChange={(event) => setOnset(event.target.value)}
                    maxLength={80}
                    placeholder="Today / 3 days / unsure"
                    className="mt-2 h-11 w-full rounded-xl border border-border bg-background px-3 text-sm"
                  />
                </label>
              </div>
              <details className="rounded-2xl border border-border/60 bg-background/50 p-4">
                <summary className="cursor-pointer text-sm font-semibold">
                  Add whole-pet context
                </summary>
                <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                  A photo can look minor while the overall pet is unwell.
                </p>
                <div className="mt-4 grid grid-cols-2 gap-3">
                  <ContextSelect label="Appetite" value={appetite} onChange={setAppetite} />
                  <ContextSelect label="Energy" value={energy} onChange={setEnergy} />
                  <ContextSelect label="Breathing" value={breathing} onChange={setBreathing} />
                  <ContextSelect label="Bathroom" value={bathroom} onChange={setBathroom} />
                </div>
              </details>
            </section>

            <section className="mt-4 rounded-3xl border border-border/60 bg-card/65 p-4">
              {cameraOpen ? (
                <div>
                  <div className="relative overflow-hidden rounded-2xl bg-black">
                    <video
                      ref={videoRef}
                      playsInline
                      muted
                      className="aspect-[4/3] w-full object-cover"
                    />
                    <div
                      className="pointer-events-none absolute inset-5 rounded-2xl border border-white/50"
                      aria-hidden="true"
                    />
                  </div>
                  <p className="mt-3 text-center text-xs text-muted-foreground">
                    Fill the frame, use even light, and keep the camera steady.
                  </p>
                  <div className="mt-3 grid grid-cols-[1fr_auto] gap-2">
                    <Button
                      className="h-12 rounded-2xl"
                      onClick={captureAndAnalyze}
                      disabled={!canAnalyze}
                    >
                      <Camera className="mr-2 h-4 w-4" /> Capture & check
                    </Button>
                    <Button
                      variant="outline"
                      size="icon"
                      className="h-12 w-12 rounded-2xl"
                      onClick={stopCamera}
                      aria-label="Close camera"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ) : imagePreview ? (
                <div>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={imagePreview}
                    alt="Pet health photo selected for screening"
                    className="max-h-[420px] w-full rounded-2xl bg-muted object-contain"
                  />
                  <div className="mt-3 grid grid-cols-2 gap-2">
                    <Button
                      className="h-11 rounded-xl"
                      disabled={!canAnalyze}
                      onClick={() => analyzeMutation.mutate(image)}
                    >
                      {analyzeMutation.isPending ? (
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      ) : (
                        <Sparkles className="mr-2 h-4 w-4" />
                      )}{' '}
                      Check photo
                    </Button>
                    <Button
                      variant="outline"
                      className="h-11 rounded-xl"
                      onClick={() => {
                        setImage(null);
                        setResult(null);
                      }}
                    >
                      <RefreshCw className="mr-2 h-4 w-4" /> New photo
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="grid grid-cols-2 gap-3">
                  <button
                    type="button"
                    onClick={() => void startCamera()}
                    className="flex min-h-28 flex-col items-center justify-center rounded-2xl border border-primary/20 bg-primary/[0.055] p-4 text-center transition hover:bg-primary/[0.08]"
                  >
                    <Camera className="h-6 w-6 text-primary" />
                    <span className="mt-2 text-sm font-semibold">Open camera</span>
                    <span className="mt-1 text-[11px] text-muted-foreground">Capture & screen</span>
                  </button>
                  <label className="flex min-h-28 cursor-pointer flex-col items-center justify-center rounded-2xl border border-border/70 bg-background p-4 text-center transition hover:border-primary/30">
                    <ImagePlus className="h-6 w-6 text-muted-foreground" />
                    <span className="mt-2 text-sm font-semibold">Upload photo</span>
                    <span className="mt-1 text-[11px] text-muted-foreground">
                      JPEG, PNG or WebP
                    </span>
                    <input
                      type="file"
                      accept="image/jpeg,image/png,image/webp"
                      className="sr-only"
                      onChange={chooseImage}
                    />
                  </label>
                </div>
              )}
              {cameraError && (
                <p
                  className="mt-3 text-center text-xs text-amber-700 dark:text-amber-300"
                  role="alert"
                >
                  {cameraError}
                </p>
              )}
              {!image && !cameraOpen && (
                <Button
                  variant="ghost"
                  className="mt-3 w-full"
                  disabled={!canAnalyze}
                  onClick={() => analyzeMutation.mutate(null)}
                >
                  <MessageCircle className="mr-2 h-4 w-4" /> Ask without a photo
                </Button>
              )}
              <div className="mt-3 flex items-start gap-2 rounded-xl bg-muted/50 p-3 text-[11px] leading-relaxed text-muted-foreground">
                <ShieldCheck className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                <span>
                  Health photos are not saved to Woof media storage. A saved check stores only the
                  derived observation and an irreversible image fingerprint.
                </span>
              </div>
            </section>
          </>
        )}

        {analyzeMutation.isPending && (
          <section
            className="mt-5 rounded-3xl border border-border/60 bg-card p-6 text-center"
            role="status"
          >
            <Loader2 className="mx-auto h-6 w-6 animate-spin text-primary" />
            <p className="mt-3 font-semibold">Reviewing the whole picture</p>
            <p className="mt-1 text-xs text-muted-foreground">
              Checking urgency, what is actually visible, and whether the image is good enough to
              use.
            </p>
          </section>
        )}
        {analyzeMutation.isError && (
          <section
            className="mt-5 rounded-2xl border border-amber-500/25 bg-amber-500/[0.06] p-4"
            role="alert"
          >
            <p className="text-sm font-semibold">Health Lens could not complete this check</p>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              If your pet seems seriously unwell, do not wait for the tool. Contact a veterinarian.
            </p>
          </section>
        )}

        {result && !analyzeMutation.isPending && (
          <section className="mt-5">
            <TriageCard assessment={result.assessment} />
            <AssessmentDetails assessment={result.assessment} />
            {result.assessment.questions.length > 0 && (
              <section className="mt-4 rounded-2xl border border-border/60 bg-card/65 p-4">
                <div className="flex items-center gap-2 text-sm font-semibold">
                  <CircleHelp className="h-4 w-4 text-primary" /> Useful follow-ups
                </div>
                <ul className="mt-3 space-y-2 text-xs leading-relaxed text-muted-foreground">
                  {result.assessment.questions.map((question) => (
                    <li key={question}>• {question}</li>
                  ))}
                </ul>
              </section>
            )}
            {followUps.map((item, index) => (
              <div key={`${item.question}-${index}`} className="mt-4 space-y-2">
                <div className="ml-10 rounded-2xl rounded-br-md bg-primary px-4 py-3 text-sm text-primary-foreground">
                  {item.question}
                </div>
                <div className="mr-6 rounded-2xl rounded-bl-md border border-border/60 bg-card p-4">
                  <p className="text-sm leading-relaxed">{item.assessment.summary}</p>
                  <p className="mt-2 text-xs font-semibold text-muted-foreground">
                    {triageMeta[item.assessment.triage].label}
                  </p>
                </div>
              </div>
            ))}
            {result.assessmentId && result.assessment.triage !== 'emergency_now' && (
              <form
                className="mt-4 flex gap-2"
                onSubmit={(event) => {
                  event.preventDefault();
                  if (followUp.trim().length >= 3) followUpMutation.mutate();
                }}
              >
                <input
                  value={followUp}
                  onChange={(event) => setFollowUp(event.target.value)}
                  maxLength={1200}
                  placeholder="Ask a follow-up…"
                  className="h-12 min-w-0 flex-1 rounded-2xl border border-border bg-card px-4 text-sm"
                />
                <Button
                  type="submit"
                  size="icon"
                  className="h-12 w-12 rounded-2xl"
                  disabled={followUp.trim().length < 3 || followUpMutation.isPending}
                  aria-label="Send follow-up"
                >
                  {followUpMutation.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <ChevronRight className="h-4 w-4" />
                  )}
                </Button>
              </form>
            )}
            <p className="mt-4 text-center text-[11px] leading-relaxed text-muted-foreground">
              {result.safety}
            </p>
          </section>
        )}

        {petId && timeline.data && timeline.data.length > 0 && (
          <details className="mt-7 rounded-3xl border border-border/60 bg-card/50 p-4">
            <summary className="cursor-pointer text-sm font-semibold">
              Recent private health timeline
            </summary>
            <div className="mt-3 space-y-2">
              {timeline.data.slice(0, 8).map((entry) => (
                <div
                  key={entry.id}
                  className="flex items-start justify-between gap-3 rounded-2xl border border-border/50 bg-background/60 p-3"
                >
                  <div className="min-w-0">
                    <p className="text-xs font-semibold">{triageMeta[entry.triage].label}</p>
                    <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-muted-foreground">
                      {entry.summary}
                    </p>
                    <p className="mt-1 text-[10px] text-muted-foreground">
                      {new Date(entry.createdAt).toLocaleDateString()}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() =>
                      void healthLensApi
                        .deleteTimelineEntry(entry.id)
                        .then(() => timeline.refetch())
                    }
                    className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl text-muted-foreground hover:bg-muted"
                    aria-label="Delete health timeline entry"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              ))}
            </div>
          </details>
        )}

        <section className="mt-5 rounded-2xl border border-border/60 bg-card/40 p-4">
          <div className="flex items-start gap-2 text-[11px] leading-relaxed text-muted-foreground">
            <ShieldCheck className="mt-0.5 h-3.5 w-3.5 shrink-0 text-primary" />
            <p>
              <strong className="text-foreground">Health Lens does not diagnose.</strong> A
              normal-looking photo cannot rule out illness, pain, internal injury, toxin exposure,
              or other serious problems. Sudden breathing trouble, collapse, ongoing seizures,
              severe bleeding, inability to urinate, toxin exposure, heat stroke, major trauma, or a
              swollen abdomen with unproductive retching need urgent veterinary evaluation.
            </p>
          </div>
        </section>
      </main>
      <BottomNav />
    </div>
  );
}

type ChangeLevel = 'normal' | 'mild-change' | 'major-change' | 'unknown';
function ContextSelect({
  label,
  value,
  onChange,
}: {
  label: string;
  value: ChangeLevel;
  onChange: (value: ChangeLevel) => void;
}) {
  return (
    <label className="text-xs font-semibold">
      {label}
      <select
        value={value}
        onChange={(event) => onChange(event.target.value as ChangeLevel)}
        className="mt-1.5 h-10 w-full rounded-xl border border-border bg-background px-2 text-xs"
      >
        <option value="unknown">Not sure</option>
        <option value="normal">Normal</option>
        <option value="mild-change">A little different</option>
        <option value="major-change">Major change</option>
      </select>
    </label>
  );
}
