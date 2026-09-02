import { Body, Controller, Delete, Get, Post, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { CurrentPushSubscriptionDto, SubscribeDto } from './dto/push-subscription.dto';
import { NotificationsService } from './notifications.service';

@ApiTags('notifications')
@ApiBearerAuth()
@Controller('notifications')
@UseGuards(JwtAuthGuard)
export class NotificationsController {
  constructor(private readonly notificationsService: NotificationsService) {}

  @Get('subscription')
  @ApiOperation({ summary: 'Read the authenticated session owner push subscription status' })
  async subscriptionStatus(@Request() req: AuthenticatedRequest) {
    return this.notificationsService.getPushSubscriptionStatus(req.user.sub);
  }

  @Post('subscribe')
  @ApiOperation({ summary: 'Subscribe the authenticated session owner to push notifications' })
  async subscribe(@Body() subscribeDto: SubscribeDto, @Request() req: AuthenticatedRequest) {
    return this.notificationsService.subscribePushNotification(
      req.user.sub,
      subscribeDto.subscription
    );
  }

  @Post('subscription/revoke')
  @ApiOperation({ summary: 'Remove only the authenticated current-browser push subscription' })
  async removeCurrent(
    @Body() current: CurrentPushSubscriptionDto,
    @Request() req: AuthenticatedRequest
  ) {
    return this.notificationsService.removeCurrentPushSubscription(
      req.user.sub,
      current.subscriptionFingerprint
    );
  }

  @Delete('unsubscribe')
  @ApiOperation({ summary: 'Remove the authenticated account push row for recovery/revocation' })
  async unsubscribe(@Request() req: AuthenticatedRequest) {
    return this.notificationsService.unsubscribePushNotification(req.user.sub);
  }
}
