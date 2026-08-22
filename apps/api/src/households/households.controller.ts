import {
  Body,
  Controller,
  Delete,
  Get,
  Param,
  Patch,
  Post,
  Request,
  UseGuards,
} from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { UpdateHouseholdDto } from './dto/household.dto';
import { HouseholdsService } from './households.service';

@ApiTags('households')
@Controller('households')
@UseGuards(JwtAuthGuard)
@ApiBearerAuth()
export class HouseholdsController {
  constructor(private readonly households: HouseholdsService) {}

  @Get('me')
  @ApiOperation({ summary: 'Get dogOS households for the authenticated user' })
  getMine(@Request() req: AuthenticatedRequest) {
    return this.households.getMine(req.user.sub);
  }

  @Patch(':id')
  @ApiOperation({ summary: 'Update a managed household' })
  update(
    @Request() req: AuthenticatedRequest,
    @Param('id') id: string,
    @Body() dto: UpdateHouseholdDto
  ) {
    return this.households.update(req.user.sub, id, dto);
  }

  @Post(':id/pets/:petId')
  @ApiOperation({ summary: 'Add one owned pet to a managed household' })
  addPet(
    @Request() req: AuthenticatedRequest,
    @Param('id') id: string,
    @Param('petId') petId: string
  ) {
    return this.households.addOwnedPet(req.user.sub, id, petId);
  }

  @Delete(':id/pets/:petId')
  @ApiOperation({ summary: 'Remove one pet from a managed household without deleting the pet' })
  removePet(
    @Request() req: AuthenticatedRequest,
    @Param('id') id: string,
    @Param('petId') petId: string
  ) {
    return this.households.removePet(req.user.sub, id, petId);
  }
}
