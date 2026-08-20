import {
  Body,
  Controller,
  Delete,
  Get,
  Param,
  Post,
  Put,
  Query,
  Request,
  UseGuards,
} from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiResponse, ApiTags } from '@nestjs/swagger';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { CreatePetDto, UpdatePetDto } from './dto/create-pet.dto';
import { PetsService } from './pets.service';

@ApiTags('pets')
@Controller('pets')
@UseGuards(JwtAuthGuard)
@ApiBearerAuth()
export class PetsController {
  constructor(private petsService: PetsService) {}

  @Post()
  @ApiOperation({ summary: 'Create a pet owned by the authenticated user' })
  @ApiResponse({ status: 201, description: 'Pet created successfully' })
  @ApiResponse({ status: 400, description: 'Invalid input' })
  async create(@Request() req: any, @Body() createPetDto: CreatePetDto) {
    return this.petsService.create(req.user.sub, createPetDto);
  }

  @Get()
  @ApiOperation({ summary: 'Get pets (paginated)' })
  @ApiResponse({ status: 200, description: 'List of pets' })
  async findAll(
    @Query('skip') skip?: number,
    @Query('take') take?: number,
    @Query('ownerId') ownerId?: string,
  ) {
    return this.petsService.findAll(skip, take, ownerId);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get pet by ID' })
  @ApiResponse({ status: 200, description: 'Pet found' })
  @ApiResponse({ status: 404, description: 'Pet not found' })
  async findOne(@Param('id') id: string) {
    return this.petsService.findById(id);
  }

  @Put(':id')
  @ApiOperation({ summary: 'Update a pet owned by the authenticated user' })
  @ApiResponse({ status: 200, description: 'Pet updated successfully' })
  @ApiResponse({ status: 404, description: 'Pet not found' })
  async update(
    @Request() req: any,
    @Param('id') id: string,
    @Body() updatePetDto: UpdatePetDto,
  ) {
    return this.petsService.updateOwned(id, req.user.sub, updatePetDto);
  }

  @Delete(':id')
  @ApiOperation({ summary: 'Delete a pet owned by the authenticated user' })
  @ApiResponse({ status: 200, description: 'Pet deleted successfully' })
  @ApiResponse({ status: 404, description: 'Pet not found' })
  async delete(@Request() req: any, @Param('id') id: string) {
    return this.petsService.deleteOwned(id, req.user.sub);
  }
}
