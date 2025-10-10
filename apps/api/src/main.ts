import { NestFactory } from '@nestjs/core';
import { ValidationPipe } from '@nestjs/common';
import { SwaggerModule, DocumentBuilder } from '@nestjs/swagger';
import { AppModule } from './app.module';
import { AllExceptionsFilter } from './common/filters/all-exceptions.filter';
import { initSentry } from './sentry';

// Initialize Sentry as early as possible
initSentry();

async function bootstrap() {
  const app = await NestFactory.create(AppModule, {
    logger: ['error', 'warn', 'log', 'debug', 'verbose'],
  });

  // Global exception filter for Sentry
  app.useGlobalFilters(new AllExceptionsFilter());

  // Global prefix
  const apiPrefix = process.env.API_PREFIX || 'api/v1';
  app.setGlobalPrefix(apiPrefix);

  // CORS
  app.enableCors({
    origin: process.env.CORS_ORIGIN?.split(',') || ['http://localhost:3000'],
    credentials: true,
  });

  // Global validation pipe
  app.useGlobalPipes(
    new ValidationPipe({
      whitelist: true,
      forbidNonWhitelisted: true,
      transform: true,
      transformOptions: {
        enableImplicitConversion: true,
      },
    }),
  );

  // Swagger documentation
  const config = new DocumentBuilder()
    .setTitle('Woof API')
    .setDescription('Pet Social Fitness Platform API - Galaxy Dark Edition')
    .setVersion('1.0')
    .addTag('auth', 'Authentication endpoints')
    .addTag('users', 'User management')
    .addTag('pets', 'Pet profiles and management')
    .addTag('activities', 'Activity tracking')
    .addTag('social', 'Social features (posts, likes, comments)')
    .addTag('meetups', 'Meetup coordination')
    .addTag('compatibility', 'Pet compatibility ML')
    .addBearerAuth()
    .build();

  const document = SwaggerModule.createDocument(app, config);
  SwaggerModule.setup('docs', app, document, {
    customSiteTitle: 'Woof API Docs',
    customCss: `
      .swagger-ui .topbar { background-color: #0B1C3D; }
      .swagger-ui .info .title { color: #6BA8FF; }
    `,
  });

  const port = process.env.PORT || 4000;
  await app.listen(port);

  console.log(`
  🐾 Woof API is running!

  🚀 Server: http://localhost:${port}
  📚 API Docs: http://localhost:${port}/docs
  🔗 GraphQL: http://localhost:${port}/graphql (coming soon)

  Environment: ${process.env.NODE_ENV || 'development'}
  `);
}

bootstrap();
