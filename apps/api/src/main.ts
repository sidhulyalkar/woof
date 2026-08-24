import { Logger, ShutdownSignal, ValidationPipe } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { NestFactory } from '@nestjs/core';
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger';
import helmet from 'helmet';
import { AppModule } from './app.module';
import { AllExceptionsFilter } from './common/filters/all-exceptions.filter';
import { requestContextMiddleware } from './common/http/request-context';
import { initSentry } from './sentry';

const bootstrapLogger = new Logger('Bootstrap');

initSentry();

async function bootstrap() {
  const isProduction = process.env.NODE_ENV === 'production';
  const app = await NestFactory.create(AppModule, {
    logger: isProduction ? ['error', 'warn', 'log'] : ['error', 'warn', 'log', 'debug'],
  });
  const configService = app.get(ConfigService);

  app.enableShutdownHooks([ShutdownSignal.SIGTERM, ShutdownSignal.SIGINT]);
  app.use(requestContextMiddleware);
  app.use(
    helmet({
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          styleSrc: ["'self'", "'unsafe-inline'"],
          scriptSrc: ["'self'"],
          imgSrc: ["'self'", 'data:', 'https:'],
        },
      },
      crossOriginEmbedderPolicy: false,
    })
  );

  app.useGlobalFilters(new AllExceptionsFilter());

  const apiPrefix = configService.get<string>('API_PREFIX') || 'api/v1';
  app.setGlobalPrefix(apiPrefix);

  const allowedOrigins = (configService.get<string>('CORS_ORIGIN') || 'http://localhost:3000')
    .split(',')
    .map((origin) => origin.trim())
    .filter(Boolean);

  app.enableCors({
    origin: (origin, callback) => {
      if (!origin || allowedOrigins.includes(origin)) {
        callback(null, true);
      } else {
        callback(new Error('Not allowed by CORS'));
      }
    },
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Request-ID'],
    exposedHeaders: ['X-Total-Count', 'X-Request-ID'],
    maxAge: 3600,
  });

  app.useGlobalPipes(
    new ValidationPipe({
      whitelist: true,
      forbidNonWhitelisted: true,
      transform: true,
      transformOptions: {
        enableImplicitConversion: true,
      },
    })
  );

  const docsEnabled = configService.get<string>('API_DOCS_ENABLED') === 'true';
  if (docsEnabled) {
    const swaggerConfig = new DocumentBuilder()
      .setTitle('Woof API')
      .setDescription(
        'Application API for Woof: pet profiles, compatibility, activity, social coordination, events, messaging, preferences, and operational integrations.'
      )
      .setVersion('1.0')
      .addTag('auth', 'Authentication endpoints')
      .addTag('users', 'User profiles')
      .addTag('pets', 'Pet profiles and management')
      .addTag('activities', 'Activity tracking')
      .addTag('social', 'Posts, likes, and comments')
      .addTag('meetups', 'Meetup coordination')
      .addTag('compatibility', 'Explainable compatibility ranking')
      .addTag('quiz', 'Matching preference sessions')
      .addBearerAuth()
      .build();

    const document = SwaggerModule.createDocument(app, swaggerConfig);
    SwaggerModule.setup('docs', app, document, {
      customSiteTitle: 'Woof API Docs',
      customCss: `
        .swagger-ui .topbar { background-color: #0d1117; }
        .swagger-ui .info .title { color: #b86912; }
      `,
    });
  }

  const port = configService.get<number>('PORT') || 4000;
  await app.listen(port);

  bootstrapLogger.log(
    `Woof API listening port=${port} api_prefix=/${apiPrefix} docs=${docsEnabled ? 'enabled' : 'disabled'} mode=${configService.get<string>('NODE_ENV') || 'development'}`
  );
}

bootstrap().catch((error: unknown) => {
  const errorName = error instanceof Error ? error.name : 'UnknownError';
  bootstrapLogger.error(`Woof API failed to start error=${errorName}`);
  process.exitCode = 1;
});
