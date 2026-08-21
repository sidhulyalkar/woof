import {
  BadRequestException,
  ForbiddenException,
  Injectable,
  NotFoundException,
} from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import {
  CreateTrainingPlanDto,
  RecordTrainingSessionDto,
  UpdateTrainingPlanStatusDto,
} from './dto/coaching.dto';

const DAY_MS = 24 * 60 * 60 * 1000;
const TRAINING_GOAL_TYPE = 'TRAINING';
const PLAN_SCHEMA_VERSION = 'woof-training-plan-v1';
const SESSION_SCHEMA_VERSION = 'woof-training-session-v1';

export type TrainingSignal = {
  attempts: number;
  successes: number;
  successRate: number;
  stressSignals: string[];
  stoppedEarly: boolean;
  difficultyLevel: number;
};

export type ProgressionDecision = {
  action: 'start' | 'hold' | 'increase' | 'decrease';
  nextLevel: number;
  headline: string;
  reason: string;
};

const STRONG_CONCERN_SIGNALS = new Set([
  'freezing',
  'cowering',
  'escape-attempt',
  'growling',
  'hiding',
  'tail-tucked',
]);

export function recommendProgression(
  sessionsNewestFirst: TrainingSignal[],
  currentLevel: number,
): ProgressionDecision {
  const level = Math.max(1, Math.min(5, Math.round(currentLevel || 1)));
  if (sessionsNewestFirst.length === 0) {
    return {
      action: 'start',
      nextLevel: level,
      headline: 'Start where success is easy',
      reason: 'Use a quiet setup, mark the behavior you want, and reward every successful repetition.',
    };
  }

  const latest = sessionsNewestFirst[0];
  const hasStrongConcern = latest.stressSignals.some((signal) => STRONG_CONCERN_SIGNALS.has(signal));
  if (latest.stoppedEarly || hasStrongConcern) {
    return {
      action: 'decrease',
      nextLevel: Math.max(1, level - 1),
      headline: 'Make the next practice easier',
      reason:
        'Stopping or lowering difficulty is useful information. Add distance, reduce distraction, or shorten the ask before trying again.',
    };
  }

  const recent = sessionsNewestFirst.slice(0, 2);
  if (
    recent.length >= 2 &&
    recent.every(
      (session) =>
        session.attempts >= 5 &&
        session.successRate >= 0.8 &&
        session.stressSignals.length === 0 &&
        !session.stoppedEarly,
    )
  ) {
    return {
      action: level < 5 ? 'increase' : 'hold',
      nextLevel: Math.min(5, level + 1),
      headline: level < 5 ? 'Add one small challenge' : 'Generalize, do not pile on difficulty',
      reason:
        level < 5
          ? 'Two comfortable sessions above the practice threshold suggest the skill is ready for one modest increase in duration, distance, or distraction.'
          : 'The skill is working in varied contexts. Keep reinforcement worthwhile and vary contexts gradually.',
    };
  }

  if (recent.length >= 2) {
    const average = recent.reduce((sum, session) => sum + session.successRate, 0) / recent.length;
    if (average < 0.6) {
      return {
        action: 'decrease',
        nextLevel: Math.max(1, level - 1),
        headline: 'Shrink the challenge',
        reason:
          'Recent success is below the coaching threshold. Make the behavior easier to earn instead of repeating a cue that is not working.',
      };
    }
  }

  return {
    action: 'hold',
    nextLevel: level,
    headline: 'Repeat this level once more',
    reason:
      'Keep the environment predictable and reinforce clean repetitions before changing difficulty.',
  };
}

type TrainingTemplate = {
  id: string;
  species: 'DOG' | 'CAT' | 'ANY';
  title: string;
  skill: string;
  objective: string;
  cue: string;
  handlerFocus: string;
  steps: string[];
  rewardExamples: string[];
  safety: string;
};

const TRAINING_TEMPLATES: TrainingTemplate[] = [
  {
    id: 'dog-check-in',
    species: 'DOG',
    title: 'Check in with me',
    skill: 'attention',
    objective: 'Build voluntary attention before asking for harder behavior.',
    cue: 'Name once, then wait',
    handlerFocus: 'Mark the instant your dog looks toward you. Avoid repeating the name.',
    steps: [
      'Begin somewhere quiet and keep a reward ready.',
      'Say the name once. When your dog looks toward you, mark the moment and reward.',
      'After several easy wins, change your position slightly instead of adding a big distraction.',
    ],
    rewardExamples: ['small food reward', 'brief toy play', 'permission to sniff'],
    safety: 'If your dog cannot disengage from the environment, create more distance rather than pulling attention back by force.',
  },
  {
    id: 'dog-hand-target',
    species: 'DOG',
    title: 'Hand target',
    skill: 'targeting',
    objective: 'Teach an easy cooperative movement that can guide future skills.',
    cue: 'Touch',
    handlerFocus: 'Present the target close enough to succeed. Mark nose contact rather than moving your hand into the dog.',
    steps: [
      'Offer an open palm a few inches from the nose.',
      'Mark any deliberate nose touch and reward immediately.',
      'Move the hand only a little farther after repeated easy touches.',
    ],
    rewardExamples: ['food reward', 'toy', 'praise followed by access to something wanted'],
    safety: 'Never push your hand into a worried dog’s face. Let the pet choose to approach the target.',
  },
  {
    id: 'dog-settle-mat',
    species: 'DOG',
    title: 'Settle on a mat',
    skill: 'settling',
    objective: 'Make a portable resting place valuable and predictable.',
    cue: 'Mat',
    handlerFocus: 'Reward calm choices on the mat before asking for longer duration.',
    steps: [
      'Place the mat in a quiet area and reward looking at or approaching it.',
      'Reward paws on the mat, then relaxed body positions as they happen.',
      'Release before restlessness and build duration in small increments.',
    ],
    rewardExamples: ['small food rewards placed on the mat', 'calm praise', 'chew or enrichment item when appropriate'],
    safety: 'The mat should remain a safe place. Do not physically hold the pet there.',
  },
  {
    id: 'dog-recall-foundation',
    species: 'DOG',
    title: 'Recall foundation',
    skill: 'recall',
    objective: 'Build a strong history of moving toward the handler after one cue.',
    cue: 'Come',
    handlerFocus: 'Use the cue only when you can make success likely, then pay generously.',
    steps: [
      'Practice indoors or in a securely enclosed low-distraction space.',
      'Say the cue once, move away invitingly, and mark movement toward you.',
      'Reward close to your body, then release back to something enjoyable when safe.',
    ],
    rewardExamples: ['high-value food', 'tug or toy', 'release to sniff or play'],
    safety: 'Do not test recall off leash in unsecured areas. Use a suitable long line or secure enclosure while generalizing.',
  },
  {
    id: 'dog-loose-leash-check-in',
    species: 'DOG',
    title: 'Loose-leash check-in',
    skill: 'walking',
    objective: 'Reinforce walking and checking in without leash pressure being the teacher.',
    cue: 'Walk together',
    handlerFocus: 'Reward slack leash and voluntary check-ins before tension develops.',
    steps: [
      'Start in a quiet stretch with enough distance from major distractions.',
      'Mark and reward moments when the leash is loose or your dog checks in.',
      'If pulling rises, change direction or create distance rather than correcting through the leash.',
    ],
    rewardExamples: ['food at your side', 'permission to sniff', 'forward movement'],
    safety: 'Choose equipment and environments that let you manage safely without painful or frightening corrections.',
  },
  {
    id: 'cat-target-touch',
    species: 'CAT',
    title: 'Target touch',
    skill: 'targeting',
    objective: 'Build a voluntary target behavior for play, enrichment, and cooperative movement.',
    cue: 'Touch',
    handlerFocus: 'Let curiosity do the work. Reward orientation and contact without moving the target into the cat.',
    steps: [
      'Present a small target at a comfortable distance.',
      'Reward looking toward it, then voluntary nose contact.',
      'Move the target a few inches only after easy repetitions.',
    ],
    rewardExamples: ['tiny food reward', 'brief wand-toy play', 'petting if your cat seeks it'],
    safety: 'Stop if the cat walks away or hides. Participation should remain voluntary.',
  },
  {
    id: 'cat-carrier-comfort',
    species: 'CAT',
    title: 'Carrier comfort',
    skill: 'cooperative-care',
    objective: 'Turn the carrier into a familiar place rather than a last-minute capture event.',
    cue: 'Carrier',
    handlerFocus: 'Reward voluntary investigation. Keep the door open until entering is comfortable.',
    steps: [
      'Leave the carrier open in a familiar room with comfortable bedding.',
      'Reward looking at, approaching, and stepping into it voluntarily.',
      'Only add tiny door movement after the cat comfortably chooses to enter.',
    ],
    rewardExamples: ['food placed near or inside the carrier', 'favorite bedding', 'play near the carrier'],
    safety: 'Do not trap a worried cat during practice. If transport is medically urgent, follow veterinary guidance instead.',
  },
  {
    id: 'any-name-game',
    species: 'ANY',
    title: 'Name game',
    skill: 'attention',
    objective: 'Pair the pet’s name with good things and voluntary orientation.',
    cue: 'Name once',
    handlerFocus: 'Reward orientation instead of using the name to interrupt or scold.',
    steps: [
      'Practice when your pet is relaxed and nearby.',
      'Say the name once and reward a glance or orientation toward you.',
      'End after a few easy repetitions and return to normal life.',
    ],
    rewardExamples: ['species-appropriate food', 'play', 'access to a preferred activity'],
    safety: 'Use only rewards your pet safely enjoys and stop if participation decreases.',
  },
];

const LEVEL_LABELS = [
  'quiet and easy',
  'small duration or distance',
  'gentle distraction',
  'real-life low-intensity context',
  'varied contexts',
] as const;

type PlanMetadata = {
  schemaVersion: string;
  templateId: string;
  skill: string;
  objective: string;
  cue: string;
  handlerFocus: string;
  steps: string[];
  rewardExamples: string[];
  safety: string;
  level: number;
  targetSuccessRate: number;
  method: 'reward-based';
  lastDecision?: ProgressionDecision;
};

@Injectable()
export class CoachingService {
  constructor(private readonly prisma: PrismaService) {}

  async getDashboard(userId: string, requestedPetId?: string) {
    const pet = requestedPetId
      ? await this.prisma.pet.findFirst({ where: { id: requestedPetId, ownerId: userId } })
      : await this.prisma.pet.findFirst({
          where: { ownerId: userId },
          orderBy: { createdAt: 'asc' },
        });

    if (!pet) {
      return {
        pet: null,
        activePlan: null,
        pausedPlans: [],
        templates: [],
        weeklyRhythm: { sessions: 0, minutes: 0 },
        methodology: this.methodology(),
        onboarding: 'Add a pet profile before starting a coaching plan.',
      };
    }

    const since = new Date(Date.now() - 60 * DAY_MS);
    const [goals, activities] = await Promise.all([
      this.prisma.mutualGoal.findMany({
        where: {
          userId,
          petId: pet.id,
          goalType: TRAINING_GOAL_TYPE,
          status: { in: ['ACTIVE', 'PAUSED'] },
        },
        orderBy: { updatedAt: 'desc' },
      }),
      this.prisma.activity.findMany({
        where: {
          userId,
          petId: pet.id,
          type: 'TRAINING',
          startedAt: { gte: since },
        },
        orderBy: { startedAt: 'desc' },
        take: 120,
      }),
    ]);

    const parsedSessions = activities
      .map((activity) => this.parseSession(activity.jointMetrics, activity.startedAt, activity.endedAt))
      .filter((session): session is NonNullable<typeof session> => session !== null);

    const plans = goals
      .map((goal) => {
        const sessions = parsedSessions.filter((session) => session.planId === goal.id);
        return this.serializePlan(goal, sessions);
      })
      .filter((plan): plan is NonNullable<typeof plan> => plan !== null);

    const activePlan = plans.find((plan) => plan.status === 'ACTIVE') ?? null;
    const lastSevenDays = parsedSessions.filter(
      (session) => Date.now() - session.startedAt.getTime() <= 7 * DAY_MS,
    );

    return {
      pet: {
        id: pet.id,
        name: pet.name,
        species: pet.species,
        avatarUrl: pet.avatarUrl,
      },
      activePlan,
      pausedPlans: plans.filter((plan) => plan.status === 'PAUSED'),
      templates: this.templatesForSpecies(pet.species),
      weeklyRhythm: {
        sessions: lastSevenDays.length,
        minutes: Math.round(
          lastSevenDays.reduce((sum, session) => sum + session.durationSeconds, 0) / 60,
        ),
      },
      methodology: this.methodology(),
    };
  }

  async createPlan(userId: string, dto: CreateTrainingPlanDto) {
    const pet = await this.prisma.pet.findFirst({ where: { id: dto.petId, ownerId: userId } });
    if (!pet) throw new ForbiddenException('You do not have access to this pet');

    const template = TRAINING_TEMPLATES.find(
      (candidate) =>
        candidate.id === dto.templateId &&
        (candidate.species === 'ANY' || candidate.species === pet.species.toUpperCase()),
    );
    if (!template) throw new BadRequestException('Training template is not available for this pet');

    await this.prisma.mutualGoal.updateMany({
      where: {
        userId,
        petId: pet.id,
        goalType: TRAINING_GOAL_TYPE,
        status: 'ACTIVE',
      },
      data: { status: 'PAUSED' },
    });

    const now = new Date();
    const endDate = new Date(now.getTime() + 30 * DAY_MS);
    const metadata: PlanMetadata = {
      schemaVersion: PLAN_SCHEMA_VERSION,
      templateId: template.id,
      skill: template.skill,
      objective: template.objective,
      cue: template.cue,
      handlerFocus: template.handlerFocus,
      steps: template.steps,
      rewardExamples: template.rewardExamples,
      safety: template.safety,
      level: 1,
      targetSuccessRate: 0.8,
      method: 'reward-based',
      lastDecision: recommendProgression([], 1),
    };

    const goal = await this.prisma.mutualGoal.create({
      data: {
        userId,
        petId: pet.id,
        goalType: TRAINING_GOAL_TYPE,
        period: 'CUSTOM',
        targetNumber: 12,
        targetUnit: 'practice sessions',
        progress: 0,
        currentValue: 0,
        status: 'ACTIVE',
        startDate: now,
        endDate,
        isRecurring: false,
        metadata: metadata as unknown as Prisma.InputJsonValue,
      },
    });

    return this.serializePlan(goal, []);
  }

  async setPlanStatus(userId: string, planId: string, dto: UpdateTrainingPlanStatusDto) {
    const plan = await this.getOwnedPlan(userId, planId);
    if (dto.status === 'ACTIVE') {
      await this.prisma.mutualGoal.updateMany({
        where: {
          userId,
          petId: plan.petId,
          goalType: TRAINING_GOAL_TYPE,
          status: 'ACTIVE',
          id: { not: plan.id },
        },
        data: { status: 'PAUSED' },
      });
    }
    return this.prisma.mutualGoal.update({
      where: { id: plan.id },
      data: { status: dto.status },
    });
  }

  async recordSession(userId: string, planId: string, dto: RecordTrainingSessionDto) {
    if (dto.successes > dto.attempts) {
      throw new BadRequestException('Successes cannot exceed attempts');
    }

    const plan = await this.getOwnedPlan(userId, planId);
    if (plan.status !== 'ACTIVE') {
      throw new BadRequestException('Resume this coaching plan before logging practice');
    }
    const metadata = this.parsePlanMetadata(plan.metadata);
    if (!metadata) throw new BadRequestException('Training plan metadata is invalid');

    const recentActivities = await this.prisma.activity.findMany({
      where: {
        userId,
        petId: plan.petId,
        type: 'TRAINING',
        startedAt: { gte: new Date(Date.now() - 30 * DAY_MS) },
      },
      orderBy: { startedAt: 'desc' },
      take: 30,
    });
    const previous = recentActivities
      .map((activity) => this.parseSession(activity.jointMetrics, activity.startedAt, activity.endedAt))
      .filter(
        (session): session is NonNullable<typeof session> =>
          session !== null && session.planId === plan.id,
      );

    const now = new Date();
    const startedAt = new Date(now.getTime() - dto.durationSeconds * 1000);
    const successRate = dto.attempts > 0 ? dto.successes / dto.attempts : 0;
    const signal: TrainingSignal = {
      attempts: dto.attempts,
      successes: dto.successes,
      successRate,
      stressSignals: dto.stressSignals ?? [],
      stoppedEarly: dto.stoppedEarly ?? false,
      difficultyLevel: metadata.level,
    };
    const decision = recommendProgression(
      [signal, ...previous.map((session) => session.signal)],
      metadata.level,
    );

    const activity = await this.prisma.activity.create({
      data: {
        userId,
        petId: plan.petId,
        startedAt,
        endedAt: now,
        type: 'TRAINING',
        humanMetrics: {
          coachPlanId: plan.id,
          handlerFocus: metadata.handlerFocus,
          rewardType: dto.rewardType,
        },
        petMetrics: {
          attempts: dto.attempts,
          successes: dto.successes,
          successRate: this.round(successRate),
          stressSignals: dto.stressSignals ?? [],
          stoppedEarly: dto.stoppedEarly ?? false,
        },
        jointMetrics: {
          schemaVersion: SESSION_SCHEMA_VERSION,
          coachPlanId: plan.id,
          templateId: metadata.templateId,
          difficultyLevel: metadata.level,
          distractionLevel: dto.distractionLevel ?? metadata.level,
          attempts: dto.attempts,
          successes: dto.successes,
          successRate: this.round(successRate),
          rewardType: dto.rewardType,
          stressSignals: dto.stressSignals ?? [],
          stoppedEarly: dto.stoppedEarly ?? false,
          notes: dto.notes ?? null,
          nextDecision: decision,
        },
      },
    });

    const currentValue = plan.currentValue + 1;
    const updatedMetadata: PlanMetadata = {
      ...metadata,
      level: decision.nextLevel,
      lastDecision: decision,
    };
    const updatedPlan = await this.prisma.mutualGoal.update({
      where: { id: plan.id },
      data: {
        currentValue,
        progress: Math.min(100, (currentValue / plan.targetNumber) * 100),
        metadata: updatedMetadata as unknown as Prisma.InputJsonValue,
      },
    });

    const allSignals = [signal, ...previous.map((session) => session.signal)];
    return {
      activityId: activity.id,
      plan: this.serializePlan(updatedPlan, [
        {
          planId: plan.id,
          startedAt,
          durationSeconds: dto.durationSeconds,
          signal,
        },
        ...previous,
      ]),
      decision,
      support: this.supportMessage(allSignals),
    };
  }

  private async getOwnedPlan(userId: string, planId: string) {
    const plan = await this.prisma.mutualGoal.findUnique({ where: { id: planId } });
    if (!plan) throw new NotFoundException('Coaching plan not found');
    if (plan.userId !== userId || plan.goalType !== TRAINING_GOAL_TYPE) {
      throw new ForbiddenException('You do not have access to this coaching plan');
    }
    return plan;
  }

  private parsePlanMetadata(value: Prisma.JsonValue | null): PlanMetadata | null {
    if (!value || Array.isArray(value) || typeof value !== 'object') return null;
    const raw = value as Prisma.JsonObject;
    if (
      raw.schemaVersion !== PLAN_SCHEMA_VERSION ||
      typeof raw.templateId !== 'string' ||
      typeof raw.skill !== 'string' ||
      typeof raw.objective !== 'string' ||
      typeof raw.cue !== 'string' ||
      typeof raw.handlerFocus !== 'string' ||
      typeof raw.level !== 'number'
    ) {
      return null;
    }
    const template = TRAINING_TEMPLATES.find((candidate) => candidate.id === raw.templateId);
    if (!template) return null;
    return {
      schemaVersion: PLAN_SCHEMA_VERSION,
      templateId: raw.templateId,
      skill: raw.skill,
      objective: raw.objective,
      cue: raw.cue,
      handlerFocus: raw.handlerFocus,
      steps: template.steps,
      rewardExamples: template.rewardExamples,
      safety: template.safety,
      level: Math.max(1, Math.min(5, Math.round(raw.level))),
      targetSuccessRate: typeof raw.targetSuccessRate === 'number' ? raw.targetSuccessRate : 0.8,
      method: 'reward-based',
      lastDecision:
        raw.lastDecision && typeof raw.lastDecision === 'object' && !Array.isArray(raw.lastDecision)
          ? (raw.lastDecision as unknown as ProgressionDecision)
          : undefined,
    };
  }

  private parseSession(
    value: Prisma.JsonValue | null,
    startedAt: Date,
    endedAt: Date | null,
  ) {
    if (!value || Array.isArray(value) || typeof value !== 'object') return null;
    const raw = value as Prisma.JsonObject;
    if (
      raw.schemaVersion !== SESSION_SCHEMA_VERSION ||
      typeof raw.coachPlanId !== 'string' ||
      typeof raw.attempts !== 'number' ||
      typeof raw.successes !== 'number' ||
      typeof raw.successRate !== 'number' ||
      typeof raw.difficultyLevel !== 'number'
    ) {
      return null;
    }
    const stressSignals = Array.isArray(raw.stressSignals)
      ? raw.stressSignals.filter((signal): signal is string => typeof signal === 'string')
      : [];
    return {
      planId: raw.coachPlanId,
      startedAt,
      durationSeconds: endedAt
        ? Math.max(0, Math.round((endedAt.getTime() - startedAt.getTime()) / 1000))
        : 0,
      signal: {
        attempts: raw.attempts,
        successes: raw.successes,
        successRate: raw.successRate,
        stressSignals,
        stoppedEarly: raw.stoppedEarly === true,
        difficultyLevel: raw.difficultyLevel,
      } satisfies TrainingSignal,
    };
  }

  private serializePlan(
    goal: {
      id: string;
      petId: string;
      status: string;
      currentValue: number;
      progress: number;
      targetNumber: number;
      metadata: Prisma.JsonValue | null;
      updatedAt: Date;
    },
    sessions: Array<{
      planId: string;
      startedAt: Date;
      durationSeconds: number;
      signal: TrainingSignal;
    }>,
  ) {
    const metadata = this.parsePlanMetadata(goal.metadata);
    if (!metadata) return null;
    const signals = sessions.map((session) => session.signal);
    const decision = recommendProgression(signals, metadata.level);
    const recent = sessions.slice(0, 5);
    const totalAttempts = recent.reduce((sum, session) => sum + session.signal.attempts, 0);
    const totalSuccesses = recent.reduce((sum, session) => sum + session.signal.successes, 0);

    return {
      id: goal.id,
      petId: goal.petId,
      status: goal.status,
      title: TRAINING_TEMPLATES.find((template) => template.id === metadata.templateId)?.title ?? metadata.skill,
      skill: metadata.skill,
      objective: metadata.objective,
      cue: metadata.cue,
      handlerFocus: metadata.handlerFocus,
      steps: metadata.steps,
      rewardExamples: metadata.rewardExamples,
      safety: metadata.safety,
      level: metadata.level,
      levelLabel: LEVEL_LABELS[metadata.level - 1],
      targetSuccessRate: metadata.targetSuccessRate,
      sessionsCompleted: goal.currentValue,
      practiceCoverage: Math.round(goal.progress),
      recentSuccessRate:
        totalAttempts > 0 ? this.round(totalSuccesses / totalAttempts) : null,
      lastPracticedAt: sessions[0]?.startedAt.toISOString() ?? null,
      nextPractice: decision,
      support: this.supportMessage(signals),
      updatedAt: goal.updatedAt.toISOString(),
    };
  }

  private supportMessage(signals: TrainingSignal[]) {
    const recent = signals.slice(0, 3);
    const concerningSessions = recent.filter((session) =>
      session.stressSignals.some((signal) => STRONG_CONCERN_SIGNALS.has(signal)),
    ).length;
    if (concerningSessions >= 2) {
      return {
        recommended: true,
        message:
          'Pause this exercise if fear, aggression, pain, or distress seems persistent or escalating. A veterinarian or qualified reward-based behavior professional can help assess what is driving it.',
      };
    }
    return {
      recommended: false,
      message:
        'Woof coaches everyday skills, not diagnosis or treatment of behavior disorders. Persistent fear, aggression, pain, or panic deserves professional assessment.',
    };
  }

  private templatesForSpecies(species: string) {
    const normalized = species.toUpperCase();
    return TRAINING_TEMPLATES.filter(
      (template) => template.species === 'ANY' || template.species === normalized,
    );
  }

  private methodology() {
    return {
      version: 'reward-based-coaching-v1',
      principles: [
        'reinforce behavior you want to see again',
        'make new behavior easy before adding difficulty',
        'change one dimension of difficulty at a time',
        'treat stress and disengagement as information, not disobedience',
        'teach the human timing and setup, not just the pet response',
        'do not use pain, fear, intimidation, or forced exposure as training tools',
      ],
      progressionPolicy:
        'A level may increase after two comfortable sessions with at least five attempts and at least 80% success. Difficulty decreases when success repeatedly falls below 60% or strong concern signals appear. These are product heuristics, not clinical thresholds.',
      sources: [
        {
          label: 'AVSAB Humane Dog Training Position Statement',
          url: 'https://avsab.org/resources/position-statements/',
        },
        {
          label: 'AAHA Behavior Management Guidelines',
          url: 'https://www.aaha.org/resources/2015-aaha-canine-and-feline-behavior-management-guidelines/',
        },
      ],
    };
  }

  private round(value: number) {
    return Math.round(value * 1000) / 1000;
  }
}
