'use client';

import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Search, MapPin, Phone, Globe, Navigation, Star, Scissors, Stethoscope, Home as HomeIcon, ShoppingBag } from 'lucide-react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';
import { toast } from 'sonner';

interface Business {
  id: string;
  name: string;
  type: string;
  description: string;
  address: string;
  lat?: number;
  lng?: number;
  phone?: string;
  website?: string;
  hours?: any;
  services: string[];
  photos: string[];
  createdAt: string;
}

export function ServicesHubScreen() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedType, setSelectedType] = useState<string>('all');
  const queryClient = useQueryClient();

  const { data: businesses, isLoading } = useQuery<Business[]>({
    queryKey: ['businesses'],
    queryFn: () => apiClient.get<Business[]>('/services/businesses'),
  });

  const trackIntentMutation = useMutation({
    mutationFn: ({ businessId, action }: { businessId: string; action: string }) =>
      apiClient.post('/services/intents', { businessId, action }),
    onError: (error: any) => {
      console.error('Failed to track intent:', error);
    },
  });

  const handleAction = (businessId: string, action: 'view' | 'tap_call' | 'tap_directions' | 'tap_website' | 'tap_book') => {
    trackIntentMutation.mutate({ businessId, action });

    switch (action) {
      case 'tap_call':
        toast.success('Opening phone...');
        break;
      case 'tap_directions':
        toast.success('Opening maps...');
        break;
      case 'tap_website':
        toast.success('Opening website...');
        break;
      case 'tap_book':
        toast.success('Tracked your interest! We\'ll follow up in 24h to see if you booked.');
        break;
    }
  };

  const businessTypes = [
    { value: 'all', label: 'All', icon: ShoppingBag },
    { value: 'groomer', label: 'Groomers', icon: Scissors },
    { value: 'vet', label: 'Vets', icon: Stethoscope },
    { value: 'boarding', label: 'Boarding', icon: HomeIcon },
  ];

  const filteredBusinesses = businesses?.filter(b => {
    const matchesSearch = !searchQuery ||
      b.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      b.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesType = selectedType === 'all' || b.type === selectedType;
    return matchesSearch && matchesType;
  }) || [];

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <ShoppingBag className="w-12 h-12 mx-auto mb-4 text-accent animate-pulse" />
          <p className="text-muted-foreground">Loading services...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="p-4 border-b border-border/20">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-2xl font-bold mb-1">Services Hub</h1>
          <p className="text-sm text-muted-foreground mb-4">
            Find trusted groomers, vets, boarding & more
          </p>

          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
            <Input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search services..."
              className="pl-10"
            />
          </div>
        </div>
      </div>

      {/* Type Filter */}
      <div className="border-b border-border/20 px-4 py-3">
        <div className="max-w-4xl mx-auto">
          <div className="flex gap-2 overflow-x-auto">
            {businessTypes.map(({ value, label, icon: Icon }) => (
              <Button
                key={value}
                variant={selectedType === value ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedType(value)}
                className="gap-2 flex-shrink-0"
              >
                <Icon className="w-4 h-4" />
                {label}
              </Button>
            ))}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-4">
        <div className="max-w-4xl mx-auto space-y-4">
          {filteredBusinesses.length === 0 ? (
            <Card className="p-8 text-center">
              <ShoppingBag className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
              <h3 className="text-lg font-semibold mb-2">No services found</h3>
              <p className="text-muted-foreground">Try adjusting your search or filters</p>
            </Card>
          ) : (
            filteredBusinesses.map(business => (
              <BusinessCard
                key={business.id}
                business={business}
                onAction={(action) => handleAction(business.id, action)}
              />
            ))
          )}
        </div>
      </div>
    </div>
  );
}

interface BusinessCardProps {
  business: Business;
  onAction: (action: 'view' | 'tap_call' | 'tap_directions' | 'tap_website' | 'tap_book') => void;
}

function BusinessCard({ business, onAction }: BusinessCardProps) {
  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'groomer':
        return <Scissors className="w-4 h-4" />;
      case 'vet':
        return <Stethoscope className="w-4 h-4" />;
      case 'boarding':
        return <HomeIcon className="w-4 h-4" />;
      default:
        return <ShoppingBag className="w-4 h-4" />;
    }
  };

  return (
    <Card className="p-6 hover:shadow-lg transition-all">
      <div className="flex gap-4">
        {/* Photo */}
        {business.photos && business.photos.length > 0 ? (
          <div className="flex-shrink-0 w-24 h-24 rounded-lg overflow-hidden bg-muted">
            <img
              src={business.photos[0]}
              alt={business.name}
              className="w-full h-full object-cover"
            />
          </div>
        ) : (
          <div className="flex-shrink-0 w-24 h-24 rounded-lg bg-accent/10 flex items-center justify-center">
            {getTypeIcon(business.type)}
          </div>
        )}

        <div className="flex-1 min-w-0">
          {/* Header */}
          <div className="flex items-start justify-between gap-2 mb-2">
            <div>
              <h3 className="text-lg font-bold">{business.name}</h3>
              <Badge variant="secondary" className="text-xs mt-1">
                {getTypeIcon(business.type)}
                <span className="ml-1 capitalize">{business.type}</span>
              </Badge>
            </div>
          </div>

          {/* Description */}
          <p className="text-sm text-muted-foreground line-clamp-2 mb-3">
            {business.description}
          </p>

          {/* Services */}
          {business.services && business.services.length > 0 && (
            <div className="flex flex-wrap gap-1 mb-3">
              {business.services.slice(0, 3).map((service, idx) => (
                <Badge key={idx} variant="outline" className="text-xs">
                  {service}
                </Badge>
              ))}
              {business.services.length > 3 && (
                <Badge variant="outline" className="text-xs">
                  +{business.services.length - 3} more
                </Badge>
              )}
            </div>
          )}

          {/* Contact Info */}
          <div className="space-y-1 mb-4">
            {business.address && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <MapPin className="w-3 h-3" />
                <span className="truncate">{business.address}</span>
              </div>
            )}
            {business.phone && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Phone className="w-3 h-3" />
                <span>{business.phone}</span>
              </div>
            )}
          </div>

          {/* Actions */}
          <div className="flex flex-wrap gap-2">
            <Button
              size="sm"
              variant="outline"
              className="gap-2"
              onClick={() => {
                if (business.phone) {
                  window.location.href = `tel:${business.phone}`;
                  onAction('tap_call');
                }
              }}
              disabled={!business.phone}
            >
              <Phone className="w-3 h-3" />
              Call
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="gap-2"
              onClick={() => {
                if (business.lat && business.lng) {
                  window.open(`https://maps.google.com/?q=${business.lat},${business.lng}`, '_blank');
                  onAction('tap_directions');
                }
              }}
              disabled={!business.lat || !business.lng}
            >
              <Navigation className="w-3 h-3" />
              Directions
            </Button>
            {business.website && (
              <Button
                size="sm"
                variant="outline"
                className="gap-2"
                onClick={() => {
                  window.open(business.website, '_blank');
                  onAction('tap_website');
                }}
              >
                <Globe className="w-3 h-3" />
                Website
              </Button>
            )}
            <Button
              size="sm"
              className="gap-2"
              onClick={() => onAction('tap_book')}
            >
              <Star className="w-3 h-3 fill-current" />
              Book
            </Button>
          </div>
        </div>
      </div>
    </Card>
  );
}
