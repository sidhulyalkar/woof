'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  ArrowLeft,
  Brain,
  Camera,
  CheckCircle2,
  Film,
  Loader2,
  PauseCircle,
  ShieldCheck,
  Sparkles,
  Upload,
  Video,
  X,
} from 'lucide-react';
import Link from 'next/link';
import { useEffect, useMemo, useRef, useState } from 'react';
import { BottomNav } from '@/components/bottom-nav';
import { Button } from '@/components/ui/button';
import {
  type BehaviorContext,
  type BehaviorVisionResult,
  type HandlerAction,
  behaviorVisionApi,
} from '@/lib/api/behavior-vision';
import { cn } from '@/lib/utils';
import { useSessionStore } from '@/store/session';

const contexts: Array<{ value: BehaviorContext; label: string }> = [
  { value: 'home', label: 'Home' },
  { value: 'street', label: 'Street' },
  { value: 'park', label: 'Park' },
  { value: 'dog-park', label: 'Dog park edge' },
  { value: 'trail', label: 'Trail' },
  { value: 'training-class', label: 'Training class' },
  { value: 'daycare', label: 'Daycare' },
  { value: 'vet', label: 'Vet' },
  { value: 'vehicle', label: 'Vehicle' },
  { value: 'other', label: 'Other' },
];

const handlerActions: Array<{ value: HandlerAction; label: string }> = [
  { value: 'none', label: 'No change yet' },
  { value: 'increase-distance', label: 'Added distance' },
  { value: 'loosen-leash', label: 'Added leash slack' },
  { value: 'single-cue', label: 'One cue' },
  { value: 'repeated-cues', label: 'Repeated cues' },
  { value: 'find-it', label: 'Find-it / food scatter' },
  { value: 'parallel-walk', label: 'Parallel movement' },
  { value: 'u-turn', label: 'U-turn' },
  { value: 'pause-and-observe', label: 'Paused and observed' },
  { value: 'tighten-leash', label: 'Held leash tighter' },
  { value: 'decrease-distance', label: 'Moved closer' },
  { value: 'end-interaction', label: 'Ended interaction' },
  { value: 'other', label: 'Something else' },
];

const dimensionLabels: Record<string, string> = {
  arousal: 'Activation',
  'body-tension': 'Body tension',
  'social-orientation': 'Orientation to dog',
  'approach-tendency': 'Approach movement',
  'avoidance-tendency': 'Avoidance movement',
  'handler-engagement': 'Handler engagement',
  'environment-engagement': 'Environment engagement',
  recovery: 'Recovery',
};

function makeSessionKey() {
  return `observe-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

export default function BehaviorObservePage() {
  const user = useSessionStore((state) => state.user);
  const pets = user?.pets ?? [];
  const [petId, setPetId] = useState(pets[0]?.id ?? '');
  const [context, setContext] = useState<BehaviorContext>('street');
  const [otherDogsPresent, setOtherDogsPresent] = useState(true);
  const [leashState, setLeashState] = useState<'off-leash' | 'loose' | 'tight' | 'unknown'>(
    'unknown'
  );
  const [phase, setPhase] = useState<'baseline' | 'during-intervention' | 'recovery'>('baseline');
  const [handlerAction, setHandlerAction] = useState<HandlerAction>('none');
  const [sessionKey, setSessionKey] = useState(makeSessionKey);
  const [ownerNote, setOwnerNote] = useState('');
  const [question, setQuestion] = useState('');
  const [media, setMedia] = useState<File | Blob | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<BehaviorVisionResult | null>(null);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [recording, setRecording] = useState(false);
  const [includeAudio, setIncludeAudio] = useState(false);
  const [captureError, setCaptureError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const queryClient = useQueryClient();

  useEffect(() => {
    if (!petId && pets[0]) setPetId(pets[0].id);
  }, [petId, pets]);

  useEffect(() => {
    if (!media) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(media);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [media]);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  const profileQuery = useQuery({
    queryKey: ['behavior-profile', petId],
    queryFn: () => behaviorVisionApi.profile(petId),
    enabled: Boolean(petId),
  });

  const analyzeMutation = useMutation({
    mutationFn: async () => {
      if (!petId || !media) throw new Error('Choose a pet and add a photo or video first.');
      return behaviorVisionApi.analyze({
        petId,
        media,
        context,
        sessionKey,
        phase,
        handlerAction,
        leashState,
        otherDogsPresent,
        includeAudio,
        ownerNote: ownerNote.trim() || undefined,
        question: question.trim() || undefined,
        saveToTimeline: true,
      });
    },
    onSuccess: (next) => {
      setResult(next);
      void queryClient.invalidateQueries({ queryKey: ['behavior-profile', petId] });
      void queryClient.invalidateQueries({ queryKey: ['behavior-timeline', petId] });
    },
  });

  const feedbackMutation = useMutation({
    mutationFn: ({ accurate }: { accurate: boolean }) => {
      if (!result?.observationId) throw new Error('This observation was not saved.');
      return behaviorVisionApi.feedback(result.observationId, accurate);
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['behavior-profile', petId] });
      void queryClient.invalidateQueries({ queryKey: ['behavior-timeline', petId] });
    },
  });

  const selectedPet = pets.find((pet) => pet.id === petId) ?? null;
  const isVideo = Boolean(media?.type.startsWith('video/'));
  const recordingSupported = typeof MediaRecorder !== 'undefined';

  const releaseCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOpen(false);
    setRecording(false);
  };

  const cancelCamera = () => {
    const recorder = recorderRef.current;
    if (recorder?.state === 'recording') {
      recorder.onstop = null;
      recorder.stop();
    }
    recorderRef.current = null;
    chunksRef.current = [];
    releaseCamera();
  };

  const openCamera = async () => {
    setCaptureError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setCaptureError(
        'This browser does not expose camera capture. Upload a photo or video instead.'
      );
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' } },
        audio: includeAudio,
      });
      streamRef.current = stream;
      setCameraOpen(true);
      requestAnimationFrame(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          void videoRef.current.play();
        }
      });
    } catch (error) {
      setCaptureError(
        error instanceof Error && error.name === 'NotAllowedError'
          ? 'Camera permission was not granted. You can still upload existing media.'
          : 'Woof could not start the camera. Try again or upload existing media.'
      );
    }
  };

  const capturePhoto = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) {
      setCaptureError('The camera is not ready yet. Try again in a moment.');
      return;
    }
    const canvas = document.createElement('canvas');
    const maxWidth = 1280;
    const scale = Math.min(1, maxWidth / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    const context2d = canvas.getContext('2d');
    if (!context2d) {
      setCaptureError('This browser could not capture a still image.');
      return;
    }
    context2d.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          setCaptureError('Woof could not encode the captured photo.');
          return;
        }
        setMedia(blob);
        setResult(null);
        releaseCamera();
      },
      'image/jpeg',
      0.88
    );
  };

  const toggleRecording = () => {
    const stream = streamRef.current;
    if (!stream) return;
    if (typeof MediaRecorder === 'undefined') {
      setCaptureError(
        'Video recording is not supported in this browser. Use a photo or file upload.'
      );
      return;
    }

    if (recorderRef.current?.state === 'recording') {
      recorderRef.current.stop();
      return;
    }

    chunksRef.current = [];
    const preferred = MediaRecorder.isTypeSupported('video/webm;codecs=vp9,opus')
      ? 'video/webm;codecs=vp9,opus'
      : 'video/webm';
    const recorder = new MediaRecorder(stream, { mimeType: preferred });
    recorderRef.current = recorder;
    recorder.ondataavailable = (event) => {
      if (event.data.size) chunksRef.current.push(event.data);
    };
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: 'video/webm' });
      if (blob.size) {
        setMedia(blob);
        setResult(null);
      } else {
        setCaptureError('The recorded clip was empty. Try a shorter clip or upload a file.');
      }
      recorderRef.current = null;
      chunksRef.current = [];
      releaseCamera();
    };
    recorder.start(500);
    setRecording(true);
    window.setTimeout(() => {
      if (recorder.state === 'recording') recorder.stop();
    }, 20_000);
  };

  const startPairedObservation = () => {
    setSessionKey(makeSessionKey());
    setPhase('baseline');
    setHandlerAction('none');
    setMedia(null);
    setResult(null);
    setCaptureError(null);
  };

  const advanceToRecovery = () => {
    setPhase('recovery');
    setMedia(null);
    setResult(null);
    setCaptureError(null);
  };

  const profile = result?.profile ?? profileQuery.data;
  const usefulEffects = useMemo(
    () =>
      profile?.interventionEffects
        .filter((effect) => effect.pairedSessions >= 2 && effect.confidence >= 0.25)
        .slice(0, 3) ?? [],
    [profile]
  );

  return (
    <div className="min-h-screen bg-background pb-28">
      <header className="sticky top-0 z-20 border-b border-border/50 bg-background/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-xl items-center gap-3 px-4 py-4">
          <Button asChild variant="ghost" size="icon">
            <Link href="/coach" aria-label="Back to Coach">
              <ArrowLeft className="h-5 w-5" aria-hidden="true" />
            </Link>
          </Button>
          <div className="min-w-0 flex-1">
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-primary">
              Woof Coach
            </p>
            <h1 className="truncate text-lg font-bold">Observe together</h1>
          </div>
          <div className="rounded-full border border-border/60 bg-card px-3 py-1 text-xs font-medium text-muted-foreground">
            media is transient
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-xl space-y-5 px-4 py-5">
        <section className="rounded-3xl border border-primary/15 bg-primary/[0.055] p-5">
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-primary/10 text-primary">
              <Brain className="h-5 w-5" aria-hidden="true" />
            </div>
            <div>
              <h2 className="font-semibold">Learn this dog, not a stereotype</h2>
              <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                Capture short comparable moments. Woof separates visible behavior from possible
                explanations, then learns which handler choices are associated with better recovery
                for this individual pet.
              </p>
            </div>
          </div>
        </section>

        {pets.length > 1 && (
          <section>
            <label htmlFor="behavior-pet" className="text-sm font-semibold">
              Pet
            </label>
            <select
              id="behavior-pet"
              value={petId}
              onChange={(event) => {
                setPetId(event.target.value);
                setResult(null);
              }}
              className="mt-2 h-11 w-full rounded-xl border border-border bg-card px-3 text-sm"
            >
              {pets.map((pet) => (
                <option key={pet.id} value={pet.id}>
                  {pet.name}
                </option>
              ))}
            </select>
          </section>
        )}

        <section className="rounded-3xl border border-border/60 bg-card/70 p-5">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-sm font-semibold">Paired observation</p>
              <p className="mt-1 text-xs text-muted-foreground">
                Baseline → change one thing → recovery
              </p>
            </div>
            <Button type="button" variant="outline" size="sm" onClick={startPairedObservation}>
              New pair
            </Button>
          </div>
          <div className="mt-4 grid grid-cols-3 gap-2">
            {(['baseline', 'during-intervention', 'recovery'] as const).map((value) => (
              <button
                key={value}
                type="button"
                onClick={() => setPhase(value)}
                className={cn(
                  'min-h-11 rounded-xl border px-2 text-xs font-semibold',
                  phase === value
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border/60 text-muted-foreground'
                )}
              >
                {value === 'during-intervention'
                  ? 'Change'
                  : value.charAt(0).toUpperCase() + value.slice(1)}
              </button>
            ))}
          </div>
        </section>

        <section className="grid grid-cols-2 gap-3">
          <label className="text-sm font-semibold">
            Context
            <select
              value={context}
              onChange={(event) => setContext(event.target.value as BehaviorContext)}
              className="mt-2 h-11 w-full rounded-xl border border-border bg-card px-3 text-sm font-normal"
            >
              {contexts.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
          <label className="text-sm font-semibold">
            Leash
            <select
              value={leashState}
              onChange={(event) =>
                setLeashState(event.target.value as 'off-leash' | 'loose' | 'tight' | 'unknown')
              }
              className="mt-2 h-11 w-full rounded-xl border border-border bg-card px-3 text-sm font-normal"
            >
              <option value="unknown">Unknown</option>
              <option value="loose">Loose</option>
              <option value="tight">Tight</option>
              <option value="off-leash">Off leash</option>
            </select>
          </label>
        </section>

        <section className="rounded-2xl border border-border/60 p-4">
          <label className="flex min-h-11 items-center justify-between gap-3 text-sm font-medium">
            Another dog is present
            <input
              type="checkbox"
              checked={otherDogsPresent}
              onChange={(event) => setOtherDogsPresent(event.target.checked)}
              className="h-5 w-5"
            />
          </label>
        </section>

        {phase !== 'baseline' && (
          <section>
            <label htmlFor="handler-action" className="text-sm font-semibold">
              What did you change?
            </label>
            <select
              id="handler-action"
              value={handlerAction}
              onChange={(event) => setHandlerAction(event.target.value as HandlerAction)}
              className="mt-2 h-11 w-full rounded-xl border border-border bg-card px-3 text-sm"
            >
              {handlerActions.map((action) => (
                <option key={action.value} value={action.value}>
                  {action.label}
                </option>
              ))}
            </select>
          </section>
        )}

        <section className="space-y-3">
          <div>
            <p className="text-sm font-semibold">Capture the moment</p>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              10–20 seconds is usually enough. Include the whole dog and enough surroundings to see
              movement, distance, leash state, and interaction context.
            </p>
          </div>

          <div className="rounded-2xl border border-border/60 p-4">
            <label className="flex min-h-11 items-center justify-between gap-4 text-sm font-medium">
              <span>
                Include sound
                <span className="mt-0.5 block text-xs font-normal text-muted-foreground">
                  Off by default. Useful only when vocal timing matters.
                </span>
              </span>
              <input
                type="checkbox"
                checked={includeAudio}
                disabled={cameraOpen || recording}
                onChange={(event) => setIncludeAudio(event.target.checked)}
                className="h-5 w-5"
              />
            </label>
          </div>

          {!media && !cameraOpen && (
            <div className="grid grid-cols-2 gap-3">
              <Button
                type="button"
                variant="outline"
                className="h-24 flex-col gap-2"
                onClick={openCamera}
              >
                <Camera className="h-5 w-5" aria-hidden="true" />
                Camera
              </Button>
              <label className="flex h-24 cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border border-border bg-card text-sm font-semibold hover:bg-muted/40">
                <Upload className="h-5 w-5" aria-hidden="true" />
                Upload
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp,video/mp4,video/webm,video/quicktime"
                  className="sr-only"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) {
                      setMedia(file);
                      setResult(null);
                      setCaptureError(null);
                    }
                  }}
                />
              </label>
            </div>
          )}

          {cameraOpen && (
            <div className="overflow-hidden rounded-3xl border border-border bg-black">
              <div className="relative aspect-[4/3]">
                <video ref={videoRef} muted playsInline className="h-full w-full object-cover" />
                <button
                  type="button"
                  onClick={cancelCamera}
                  className="absolute right-3 top-3 flex h-11 w-11 items-center justify-center rounded-full bg-black/60 text-white"
                  aria-label="Close camera"
                >
                  <X className="h-5 w-5" aria-hidden="true" />
                </button>
              </div>
              <div className="grid grid-cols-2 gap-3 p-3">
                <Button
                  type="button"
                  variant="secondary"
                  onClick={capturePhoto}
                  disabled={recording}
                >
                  <Camera className="mr-2 h-4 w-4" aria-hidden="true" />
                  Photo
                </Button>
                <Button type="button" onClick={toggleRecording} disabled={!recordingSupported}>
                  {recording ? (
                    <PauseCircle className="mr-2 h-4 w-4" aria-hidden="true" />
                  ) : (
                    <Video className="mr-2 h-4 w-4" aria-hidden="true" />
                  )}
                  {recording ? 'Stop' : recordingSupported ? 'Record 20s' : 'Video unavailable'}
                </Button>
              </div>
            </div>
          )}

          {media && previewUrl && (
            <div className="overflow-hidden rounded-3xl border border-border bg-card">
              <div className="relative bg-black">
                {isVideo ? (
                  <video
                    src={previewUrl}
                    controls
                    playsInline
                    className="max-h-80 w-full object-contain"
                  />
                ) : (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={previewUrl}
                    alt="Behavior observation preview"
                    className="max-h-80 w-full object-contain"
                  />
                )}
                <button
                  type="button"
                  onClick={() => {
                    setMedia(null);
                    setResult(null);
                  }}
                  className="absolute right-3 top-3 flex h-11 w-11 items-center justify-center rounded-full bg-black/60 text-white"
                  aria-label="Remove media"
                >
                  <X className="h-5 w-5" aria-hidden="true" />
                </button>
              </div>
              <div className="flex items-center gap-2 p-3 text-xs text-muted-foreground">
                {isVideo ? (
                  <Film className="h-4 w-4" aria-hidden="true" />
                ) : (
                  <Camera className="h-4 w-4" aria-hidden="true" />
                )}
                Raw media will not be added to your Woof timeline. Audio analysis is{' '}
                {includeAudio ? 'enabled for this observation' : 'disabled'}.
              </div>
            </div>
          )}

          {captureError && (
            <div
              role="alert"
              className="rounded-2xl border border-amber-500/25 bg-amber-500/[0.06] p-4 text-sm"
            >
              {captureError}
            </div>
          )}
        </section>

        <section className="space-y-3">
          <label className="block text-sm font-semibold">
            What happened?
            <textarea
              value={ownerNote}
              onChange={(event) => setOwnerNote(event.target.value)}
              maxLength={800}
              rows={3}
              placeholder="Example: another dog crossed the street; Luna started pacing and barking while I shortened the leash."
              className="mt-2 w-full rounded-2xl border border-border bg-card p-3 text-sm font-normal outline-none focus:ring-2 focus:ring-primary/20"
            />
          </label>
          <label className="block text-sm font-semibold">
            What are you trying to understand?{' '}
            <span className="font-normal text-muted-foreground">Optional</span>
            <input
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              maxLength={500}
              placeholder="Does adding distance seem to help her recover?"
              className="mt-2 h-11 w-full rounded-xl border border-border bg-card px-3 text-sm font-normal outline-none focus:ring-2 focus:ring-primary/20"
            />
          </label>
        </section>

        <Button
          type="button"
          className="h-12 w-full rounded-2xl"
          disabled={!petId || !media || analyzeMutation.isPending}
          onClick={() => analyzeMutation.mutate()}
        >
          {analyzeMutation.isPending ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
          ) : (
            <Sparkles className="mr-2 h-4 w-4" aria-hidden="true" />
          )}
          Analyze this observation
        </Button>

        {analyzeMutation.isError && (
          <div
            role="alert"
            className="rounded-2xl border border-destructive/30 bg-destructive/5 p-4 text-sm"
          >
            {analyzeMutation.error instanceof Error
              ? analyzeMutation.error.message
              : 'Behavior analysis could not be completed.'}
          </div>
        )}

        {result && (
          <section className="space-y-4 rounded-3xl border border-border/60 bg-card/75 p-5">
            <div>
              <p className="eyebrow">What Woof observed</p>
              <h2 className="mt-2 text-xl font-bold">{result.coach.headline}</h2>
              {result.coach.observableSummary && (
                <p className="mt-2 text-sm leading-relaxed text-foreground">
                  {result.coach.observableSummary}
                </p>
              )}
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                {result.coach.explanation}
              </p>
            </div>

            {result.analysis.mediaQuality.usable && result.analysis.dimensions.length > 0 && (
              <div className="space-y-3">
                {result.analysis.dimensions
                  .filter((dimension) => dimension.confidence >= 0.4)
                  .slice(0, 6)
                  .map((dimension) => (
                    <div key={dimension.dimension}>
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-medium">
                          {dimensionLabels[dimension.dimension] ?? dimension.dimension}
                        </span>
                        <span className="text-muted-foreground">
                          {Math.round(dimension.value * 100)}%
                        </span>
                      </div>
                      <div className="mt-1 h-2 overflow-hidden rounded-full bg-muted">
                        <div
                          className="h-full rounded-full bg-primary"
                          style={{ width: `${Math.round(dimension.value * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
              </div>
            )}

            {result.coach.hypothesis && (
              <div className="rounded-2xl border border-amber-500/20 bg-amber-500/[0.07] p-4">
                <p className="text-xs font-semibold uppercase tracking-wide text-amber-800 dark:text-amber-300">
                  Possible pattern · not a readout of emotion
                </p>
                <p className="mt-2 text-sm font-medium">{result.coach.hypothesis.statement}</p>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                  {result.coach.hypothesis.caveat}
                </p>
              </div>
            )}

            {result.coach.socialSafety && (
              <div className="rounded-2xl border border-border/60 p-4">
                <p className="text-sm font-semibold">About barking or pulling toward another dog</p>
                <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                  {result.coach.socialSafety}
                </p>
              </div>
            )}

            {result.coach.nextSteps.length > 0 && (
              <div>
                <p className="text-sm font-semibold">Next safe experiment</p>
                <ol className="mt-3 space-y-2">
                  {result.coach.nextSteps.map((step, index) => (
                    <li key={step} className="flex gap-3 text-sm leading-relaxed">
                      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">
                        {index + 1}
                      </span>
                      <span>{step}</span>
                    </li>
                  ))}
                </ol>
              </div>
            )}

            {result.observationId && (
              <div className="border-t border-border/60 pt-4">
                <p className="text-xs font-medium text-muted-foreground">
                  Did Woof describe the visible behavior correctly?
                </p>
                <div className="mt-2 flex gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => feedbackMutation.mutate({ accurate: true })}
                    disabled={feedbackMutation.isPending}
                  >
                    <CheckCircle2 className="mr-1.5 h-4 w-4" aria-hidden="true" /> Yes
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => feedbackMutation.mutate({ accurate: false })}
                    disabled={feedbackMutation.isPending}
                  >
                    Not quite
                  </Button>
                </div>
              </div>
            )}

            {phase === 'baseline' && (
              <Button
                type="button"
                variant="secondary"
                className="w-full"
                onClick={advanceToRecovery}
              >
                I changed one thing · record recovery
              </Button>
            )}
          </section>
        )}

        {profile && (
          <section className="rounded-3xl border border-border/60 bg-card/55 p-5">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="eyebrow">Individual model</p>
                <h2 className="mt-2 font-semibold">
                  {selectedPet?.name ?? 'Your pet'} · {profile.sampleCount} usable observations
                </h2>
              </div>
              <span className="rounded-full bg-primary/10 px-3 py-1 text-xs font-semibold text-primary">
                {Math.round(profile.personalizationConfidence * 100)}% learned
              </span>
            </div>
            <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
              Confidence grows from repeated observations across contexts and owner corrections. It
              is not a personality score.
            </p>

            {usefulEffects.length > 0 && (
              <div className="mt-4 space-y-2">
                <p className="text-sm font-semibold">Handler strategies worth testing again</p>
                {usefulEffects.map((effect) => (
                  <div
                    key={effect.action}
                    className="rounded-xl border border-border/60 p-3 text-xs"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-semibold">{effect.action}</span>
                      <span className="text-muted-foreground">
                        {effect.pairedSessions} paired clips
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        )}

        <section className="rounded-3xl border border-border/60 bg-muted/25 p-5">
          <div className="flex gap-3">
            <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-primary" aria-hidden="true" />
            <div>
              <p className="text-sm font-semibold">Behavior is contextual</p>
              <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                A video can show posture, movement, orientation, interaction and recovery. It cannot
                directly reveal an internal emotion or prove that a dog wants social contact. For
                aggression, persistent fear, panic, pain, or dangerous interactions, use a qualified
                veterinary or reward-based behavior professional rather than testing the situation.
              </p>
            </div>
          </div>
        </section>
      </main>

      <BottomNav />
    </div>
  );
}
