import { Controller, Get, Post, Put, Delete, Param, Body, Query, UseGuards } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiBearerAuth } from '@nestjs/swagger';
import { PetsService } from './pets.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { Prisma } from '@woof/database';

@ApiTags('pets')
@Controller('pets')
@UseGuards(JwtAuthGuard)
@ApiBearerAuth()
export class PetsController {
  constructor(private petsService: PetsService) {}

  @Post()
  @ApiOperation({ summary: 'Create a new pet' })
  @ApiResponse({ status: 201, description: 'Pet created successfully' })
  @ApiResponse({ status: 400, description: 'Invalid input' })
  async create(@Body() createPetDto: Prisma.PetCreateInput) {
    return this.petsService.create(createPetDto);
  }

  @Get()
  @ApiOperation({ summary: 'Get all pets (paginated)' })
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
  @ApiOperation({ summary: 'Update pet by ID' })
  @ApiResponse({ status: 200, description: 'Pet updated successfully' })
  @ApiResponse({ status: 404, description: 'Pet not found' })
  async update(
    @Param('id') id: string,
    @Body() updatePetDto: Prisma.PetUpdateInput,
  ) {
    return this.petsService.update(id, updatePetDto);
  }

  @Delete(':id')
  @ApiOperation({ summary: 'Delete pet by ID' })
  @ApiResponse({ status: 200, description: 'Pet deleted successfully' })
  @ApiResponse({ status: 404, description: 'Pet not found' })
  async delete(@Param('id') id: string) {
    return this.petsService.delete(id);
  }
}
