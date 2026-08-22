import { ValidationPipe } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { NestFactory } from '@nestjs/core';
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger';
import helmet from 'helmet';
import { AppModule } from './app.module';
import { AllExceptionsFilter } from './common/filters/all-exceptions.filter';
import { initSentry } from './sentry';

initSentry();

async function bootstrap() {
  const isProduction = process.env.NODE_ENV === 'production';
  const app = await NestFactory.create(AppModule, {
    logger: isProduction ? ['error', 'warn', 'log'] : ['error', 'warn', 'log', 'debug'],
  });
  const configService = app.get(ConfigService);

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
    allowedHeaders: ['Content-Type', 'Authorization'],
    exposedHeaders: ['X-Total-Count'],
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

  const apiDocsEnabled = !isProduction || configService.get<string>('API_DOCS_ENABLED') === 'true';
  if (apiDocsEnabled) {
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

  console.log(`
  🐾 Woof API is running
  Server: http://localhost:${port}
  API:    http://localhost:${port}/${apiPrefix}
  Docs:   ${apiDocsEnabled ? `http://localhost:${port}/docs` : 'disabled'}
  Mode:   ${configService.get<string>('NODE_ENV') || 'development'}
  `);
}

bootstrap();
