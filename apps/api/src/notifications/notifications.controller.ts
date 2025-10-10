import {
  Controller,
  Post,
  Delete,
  Body,
  UseGuards,
  Request,
  Param,
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiBearerAuth } from '@nestjs/swagger';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { NotificationsService } from './notifications.service';
import { SubscribeDto, SendPushDto } from './dto/push-subscription.dto';

@ApiTags('notifications')
@ApiBearerAuth()
@Controller('notifications')
@UseGuards(JwtAuthGuard)
export class NotificationsController {
  constructor(private readonly notificationsService: NotificationsService) {}

  @Post('subscribe')
  @ApiOperation({ summary: 'Subscribe to push notifications' })
  async subscribe(@Body() subscribeDto: SubscribeDto) {
    return this.notificationsService.subscribePushNotification(
      subscribeDto.userId,
      subscribeDto.subscription,
    );
  }

  @Delete('unsubscribe/:endpoint')
  @ApiOperation({ summary: 'Unsubscribe from push notifications' })
  async unsubscribe(@Param('endpoint') endpoint: string, @Request() req) {
    return this.notificationsService.unsubscribePushNotification(
      req.user.id,
      endpoint,
    );
  }

  @Post('send')
  @ApiOperation({ summary: 'Send a push notification (admin/testing)' })
  async sendPush(@Body() sendPushDto: SendPushDto) {
    return this.notificationsService.sendPushNotification(sendPushDto);
  }
}
