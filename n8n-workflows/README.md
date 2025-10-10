# n8n Workflow Automation Setup

This directory contains automated workflows for PetPath's user engagement, follow-ups, and reminders.

## 🚀 Quick Start

### 1. Start n8n

```bash
# Start all services including n8n
docker-compose up -d

# Check n8n logs
docker logs woof-n8n -f
```

### 2. Access n8n UI

- **URL**: http://localhost:5678
- **Username**: `admin`
- **Password**: `woofadmin`

### 3. Import Workflows

1. Navigate to **Workflows** in n8n UI
2. Click **Import from File**
3. Import each workflow JSON file from this directory:
   - `service-booking-followup.json` - 24h service booking follow-up
   - `meetup-feedback-reminder.json` - Meetup confirmation requests
   - `event-reminder-followup.json` - Event reminders and feedback
   - `fitness-goal-achievement.json` - Daily goal tracking and rewards

### 4. Configure Workflow Credentials

Each workflow requires API credentials to connect to the Woof backend:

1. Go to **Credentials** in n8n
2. Create **HTTP Request** credential:
   - **Name**: `Woof API`
   - **Authentication**: Header Auth
   - **Header Name**: `x-api-key`
   - **Header Value**: Your internal API key (set in backend env)

## 📋 Workflows Overview

### 1. Service Booking 24h Follow-up

**File**: `service-booking-followup.json`

**Trigger**: Webhook from backend when ServiceIntent created with `action='tap_book'`

**Flow**:
```
Webhook → Wait 24h → Check conversion status → Send notification → Update record
```

**Backend Integration**:
```typescript
// apps/api/src/services/services.controller.ts
async trackIntent(intentDto: TrackIntentDto) {
  const intent = await this.servicesService.trackIntent(intentDto);

  // Trigger n8n workflow
  if (intent.action === 'tap_book') {
    await fetch('http://localhost:5678/webhook/service-booking-followup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        intentId: intent.id,
        userId: intent.userId,
        serviceId: intent.serviceId,
        serviceName: intent.service.name
      })
    });
  }
}
```

### 2. Meetup Feedback Reminder

**File**: `meetup-feedback-reminder.json`

**Trigger**: Cron job runs every 30 minutes, checks for completed meetups

**Flow**:
```
Cron → Query recent meetups → Check status → Send reminder → Wait for feedback
```

**Backend Integration**:
```typescript
// n8n makes HTTP requests to:
// GET /meetups?datetime[lte]=${now}&datetime[gte]=${2hoursAgo}&status=pending
// POST /notifications/send (push notification to users)
```

### 3. Event Reminder & Follow-up

**File**: `event-reminder-followup.json`

**Trigger**: Cron job runs every hour, checks upcoming events (3h before start)

**Flow**:
```
Cron → Query events (3h away) → Get RSVPs → Send reminders → After event → Feedback request
```

**Backend Integration**:
```typescript
// n8n makes HTTP requests to:
// GET /events?datetime[gte]=${3hoursFromNow}&datetime[lte]=${3hoursFromNow+1h}
// GET /events/{eventId}/rsvps?response=yes
// POST /notifications/send (event reminder)
```

### 4. Fitness Goal Achievement

**File**: `fitness-goal-achievement.json`

**Trigger**: Daily cron at 9 PM

**Flow**:
```
Cron (9 PM) → Query users with goals → Calculate progress → Award points → Send notification
```

**Backend Integration**:
```typescript
// n8n makes HTTP requests to:
// GET /users?hasGoals=true
// GET /activities?userId={id}&startDate={weekStart}&endDate={now}
// POST /gamification/points (award achievement points)
// POST /notifications/send (congratulations notification)
```

## 🔧 Webhook Configuration

### Create Internal API Key

Add to `apps/api/.env`:
```bash
N8N_WEBHOOK_SECRET=your-random-secret-key-here
```

### Secure Webhook Endpoints

```typescript
// apps/api/src/common/guards/webhook-auth.guard.ts
@Injectable()
export class WebhookAuthGuard implements CanActivate {
  canActivate(context: ExecutionContext): boolean {
    const request = context.switchToHttp().getRequest();
    const apiKey = request.headers['x-api-key'];
    return apiKey === process.env.N8N_WEBHOOK_SECRET;
  }
}

// Apply to webhook controllers
@UseGuards(WebhookAuthGuard)
@Controller('webhooks/n8n')
export class N8nWebhooksController {
  // Webhook endpoints that n8n can call
}
```

## 📊 Monitoring & Logs

### View Workflow Executions

1. Go to **Executions** in n8n UI
2. Filter by workflow name
3. View execution details, errors, and logs

### Debugging Tips

- **Workflow not triggering**: Check cron expression and timezone
- **HTTP requests failing**: Verify API credentials and endpoint URLs
- **Webhook not receiving data**: Check network connectivity (Docker network)
- **Notifications not sending**: Verify push subscription exists for user

### Production Considerations

1. **Set proper timezone** in n8n container:
   ```yaml
   environment:
     - GENERIC_TIMEZONE=America/Los_Angeles
     - TZ=America/Los_Angeles
   ```

2. **Use production webhook URL**:
   ```yaml
   environment:
     - WEBHOOK_URL=https://your-domain.com/
   ```

3. **Enable execution logging**:
   ```yaml
   environment:
     - EXECUTIONS_DATA_SAVE_ON_SUCCESS=all
     - EXECUTIONS_DATA_SAVE_ON_ERROR=all
   ```

4. **Backup workflows** regularly by exporting to this directory

## 🧪 Testing Workflows

### Manual Testing

1. **Service Follow-up**:
   ```bash
   curl -X POST http://localhost:5678/webhook/service-booking-followup \
     -H "Content-Type: application/json" \
     -d '{
       "intentId": "test-intent-id",
       "userId": "user-id-here",
       "serviceId": "service-id-here",
       "serviceName": "Golden Gate Dog Grooming"
     }'
   ```

2. **Event Reminder**:
   - Set event datetime to 2h 50min from now
   - Wait for cron to trigger
   - Check execution logs

### Staging Environment

Before deploying to production:
1. Test all workflows in staging with real data
2. Verify notifications are sent correctly
3. Check conversion tracking updates
4. Monitor execution times and errors

## 📝 Workflow Version Control

After making changes in n8n UI:
1. Export workflow as JSON
2. Save to this directory
3. Commit to git
4. Document changes in this README

## 🔒 Security Best Practices

1. **Never commit credentials** - Use n8n credential management
2. **Rotate API keys** regularly
3. **Use HTTPS** in production
4. **Limit webhook access** with IP whitelisting if possible
5. **Monitor execution logs** for suspicious activity

## 🆘 Troubleshooting

### n8n Won't Start

```bash
# Check logs
docker logs woof-n8n

# Restart n8n
docker-compose restart n8n

# Rebuild if needed
docker-compose up -d --build n8n
```

### Database Connection Issues

```bash
# Verify postgres is running
docker ps | grep postgres

# Check n8n database
docker exec -it woof-postgres psql -U postgres -d woof_n8n -c "SELECT * FROM public.workflow_entity;"
```

### Workflows Not Executing

1. Check cron expression is valid
2. Verify workflow is **Active** (toggle in UI)
3. Check execution error logs
4. Test with manual trigger first

## 📚 Additional Resources

- [n8n Documentation](https://docs.n8n.io)
- [Webhook Nodes](https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.webhook/)
- [Cron Expression Generator](https://crontab.guru/)
- [HTTP Request Node](https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.httprequest/)
