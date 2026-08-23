import { IsLatitude, IsLongitude } from 'class-validator';

export class UpdateDiscoveryLocationDto {
  @IsLatitude()
  latitude!: number;

  @IsLongitude()
  longitude!: number;
}
