import type { ConfigContext, ExpoConfig } from 'expo/config';

const DEVELOPMENT_API_URL = 'http://localhost:4000/api/v1';
const NON_REMOTE_HOSTS = new Set(['localhost', '127.0.0.1', '0.0.0.0', '::1']);

type StaticExpoConfig = ConfigContext['config'];
type EasExtra = {
  projectId?: string;
  [key: string]: unknown;
};

function firstNonEmpty(...values: Array<string | undefined>): string | undefined {
  return values.map((value) => value?.trim()).find((value) => Boolean(value));
}

function normalizeApiUrl(value: string): string {
  return value.replace(/\/$/, '');
}

function assertRemoteApiUrl(value: string, buildProfile: string): string {
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error(`Woof ${buildProfile} API URL is not a valid absolute URL`);
  }

  if (parsed.protocol !== 'https:') {
    throw new Error(`Woof ${buildProfile} API URL must use HTTPS`);
  }

  if (NON_REMOTE_HOSTS.has(parsed.hostname.toLowerCase())) {
    throw new Error(`Woof ${buildProfile} API URL must not target a loopback host`);
  }

  return normalizeApiUrl(value);
}

function resolveBuildProfile(): string {
  return firstNonEmpty(process.env.EAS_BUILD_PROFILE, process.env.WOOF_BUILD_PROFILE) ?? 'development';
}

function resolveApiUrl(config: StaticExpoConfig, buildProfile: string): string {
  const configured = firstNonEmpty(
    process.env.EXPO_PUBLIC_API_URL,
    typeof config.extra?.apiUrl === 'string' ? config.extra.apiUrl : undefined,
  );

  if (buildProfile === 'development') {
    return normalizeApiUrl(configured ?? DEVELOPMENT_API_URL);
  }

  if (!process.env.EXPO_PUBLIC_API_URL?.trim()) {
    throw new Error(
      `Woof ${buildProfile} builds require EXPO_PUBLIC_API_URL from the selected EAS environment`,
    );
  }

  return assertRemoteApiUrl(process.env.EXPO_PUBLIC_API_URL.trim(), buildProfile);
}

function resolveProjectId(config: StaticExpoConfig, buildProfile: string): string | undefined {
  const staticEas = config.extra?.eas as EasExtra | undefined;
  const projectId = firstNonEmpty(
    process.env.EAS_PROJECT_ID,
    process.env.EAS_BUILD_PROJECT_ID,
    staticEas?.projectId,
  );

  if (projectId?.toLowerCase().includes('your-project-id')) {
    throw new Error('Woof EAS project authority is still a placeholder');
  }

  if (buildProfile !== 'development' && !projectId) {
    throw new Error(
      `Woof ${buildProfile} builds require a real EAS project id via EAS_PROJECT_ID or linked app config`,
    );
  }

  return projectId;
}

export default ({ config }: ConfigContext): ExpoConfig => {
  const buildProfile = resolveBuildProfile();
  const apiUrl = resolveApiUrl(config, buildProfile);
  const projectId = resolveProjectId(config, buildProfile);
  const existingEas = config.extra?.eas as EasExtra | undefined;

  return {
    ...config,
    name: config.name ?? 'Woof',
    slug: config.slug ?? 'woof',
    extra: {
      ...config.extra,
      apiUrl,
      buildProfile,
      ...(projectId
        ? {
            eas: {
              ...existingEas,
              projectId,
            },
          }
        : { eas: undefined }),
    },
  };
};
