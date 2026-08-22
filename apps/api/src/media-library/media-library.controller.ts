import {
  Body,
  Controller,
  Delete,
  Get,
  Param,
  Patch,
  Post,
  Query,
  Request,
  UseGuards,
} from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import {
  CompleteMediaUploadDto,
  CreateMediaAlbumDto,
  CreateMediaUploadIntentDto,
  GooglePhotosExportDto,
  GooglePhotosPickerImportDto,
  GooglePhotosPickerStartDto,
  MediaExportManifestDto,
  MediaLibraryQueryDto,
  UpdateMediaAssetDto,
} from './dto/media-library.dto';
import { MediaLibraryService } from './media-library.service';

@ApiTags('media-library')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('media-library')
export class MediaLibraryController {
  constructor(private readonly mediaLibrary: MediaLibraryService) {}

  @Post('uploads/intents')
  @ApiOperation({ summary: 'Create a short-lived direct upload URL for private pet media' })
  createUploadIntent(@Request() req: { user: { sub: string } }, @Body() dto: CreateMediaUploadIntentDto) {
    return this.mediaLibrary.createUploadIntent(req.user.sub, dto);
  }

  @Post('uploads/complete')
  @ApiOperation({ summary: 'Verify a private upload and make it visible in the pet library' })
  completeUpload(@Request() req: { user: { sub: string } }, @Body() dto: CompleteMediaUploadDto) {
    return this.mediaLibrary.completeUpload(req.user.sub, dto);
  }

  @Get()
  @ApiOperation({ summary: 'List private media, smart albums, and custom albums for one pet' })
  library(@Request() req: { user: { sub: string } }, @Query() query: MediaLibraryQueryDto) {
    return this.mediaLibrary.library(req.user.sub, query);
  }

  @Get('albums')
  @ApiOperation({ summary: 'List smart and custom albums for one pet' })
  albums(@Request() req: { user: { sub: string } }, @Query('petId') petId: string) {
    return this.mediaLibrary.albums(req.user.sub, petId);
  }

  @Post('albums')
  @ApiOperation({ summary: 'Create a private custom pet album' })
  createAlbum(@Request() req: { user: { sub: string } }, @Body() dto: CreateMediaAlbumDto) {
    return this.mediaLibrary.createAlbum(req.user.sub, dto);
  }

  @Patch('assets/:assetId')
  @ApiOperation({ summary: 'Favorite, tag, or organize one private media asset' })
  updateAsset(
    @Request() req: { user: { sub: string } },
    @Param('assetId') assetId: string,
    @Body() dto: UpdateMediaAssetDto,
  ) {
    return this.mediaLibrary.updateAsset(req.user.sub, assetId, dto);
  }

  @Delete('assets/:assetId')
  @ApiOperation({ summary: 'Delete a private media object and its library metadata' })
  deleteAsset(@Request() req: { user: { sub: string } }, @Param('assetId') assetId: string) {
    return this.mediaLibrary.deleteAsset(req.user.sub, assetId);
  }

  @Post('providers/google-photos/picker')
  @ApiOperation({ summary: 'Start a Google Photos Picker session without broad library access' })
  startGooglePhotosPicker(
    @Request() req: { user: { sub: string } },
    @Body() dto: GooglePhotosPickerStartDto,
  ) {
    return this.mediaLibrary.startGooglePhotosPicker(req.user.sub, dto);
  }

  @Post('providers/google-photos/import')
  @ApiOperation({ summary: 'Import only the media explicitly selected in a Google Photos Picker session' })
  importGooglePhotos(
    @Request() req: { user: { sub: string } },
    @Body() dto: GooglePhotosPickerImportDto,
  ) {
    return this.mediaLibrary.importGooglePhotos(req.user.sub, dto);
  }

  @Post('providers/google-photos/export')
  @ApiOperation({ summary: 'Export selected Woof media into Google Photos as app-created items' })
  exportGooglePhotos(
    @Request() req: { user: { sub: string } },
    @Body() dto: GooglePhotosExportDto,
  ) {
    return this.mediaLibrary.exportToGooglePhotos(req.user.sub, dto);
  }

  @Post('export/manifest')
  @ApiOperation({ summary: 'Create a portable manifest with short-lived original download URLs' })
  exportManifest(@Request() req: { user: { sub: string } }, @Body() dto: MediaExportManifestDto) {
    return this.mediaLibrary.exportManifest(req.user.sub, dto);
  }
}
