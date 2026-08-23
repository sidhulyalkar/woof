import { Body, Controller, Get, Param, Post, Query, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { ChatService } from './chat.service';

@ApiTags('chat')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('chat')
export class ChatController {
  constructor(private readonly chatService: ChatService) {}

  @Get('conversations')
  @ApiOperation({ summary: 'List authorized direct conversations' })
  list(@Request() req: AuthenticatedRequest) {
    return this.chatService.listConversations(req.user.sub);
  }

  @Post('conversations')
  @ApiOperation({ summary: 'Create or reuse an authorized direct conversation' })
  create(
    @Request() req: AuthenticatedRequest,
    @Body() body: { participantId?: string; participantIds?: string[] },
  ) {
    const participantId = body.participantId ?? body.participantIds?.[0];
    return this.chatService.createDirectConversation(req.user.sub, participantId ?? '');
  }

  @Get('conversations/:id/messages')
  @ApiOperation({ summary: 'Read bounded canonical message history' })
  messages(
    @Request() req: AuthenticatedRequest,
    @Param('id') id: string,
    @Query('page') page?: string,
    @Query('limit') limit?: string,
  ) {
    return this.chatService.getMessages(
      req.user.sub,
      id,
      page ? Number(page) : 1,
      limit ? Number(limit) : 50,
    );
  }

  @Post('conversations/:id/read')
  @ApiOperation({ summary: 'Advance the signed-in participant read watermark' })
  markRead(@Request() req: AuthenticatedRequest, @Param('id') id: string) {
    return this.chatService.markRead(req.user.sub, id);
  }
}
