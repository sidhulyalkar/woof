import apiClient from './client';
import { CommunityEvent, EventRSVP, CreateEventDto, PaginatedResponse } from '../types';

export const eventsApi = {
  async getEvents(page: number = 1, limit: number = 20): Promise<PaginatedResponse<CommunityEvent>> {
    return apiClient.get('/events', { params: { page, limit } });
  },

  async getEventById(id: string): Promise<CommunityEvent> {
    return apiClient.get(`/events/${id}`);
  },

  async createEvent(data: CreateEventDto): Promise<CommunityEvent> {
    return apiClient.post('/events', data);
  },

  async updateEvent(id: string, data: Partial<CreateEventDto>): Promise<CommunityEvent> {
    return apiClient.patch(`/events/${id}`, data);
  },

  async deleteEvent(id: string): Promise<void> {
    return apiClient.delete(`/events/${id}`);
  },

  async rsvpEvent(eventId: string, status: 'going' | 'maybe' | 'not_going', petIds?: string[]): Promise<EventRSVP> {
    return apiClient.post(`/events/${eventId}/rsvp`, { status, petIds });
  },

  async checkInEvent(eventId: string): Promise<void> {
    return apiClient.post(`/events/${eventId}/check-in`);
  },

  async getMyEvents(): Promise<CommunityEvent[]> {
    return apiClient.get('/events/my-events');
  },

  async getNearbyEvents(latitude: number, longitude: number, radius: number = 10000): Promise<CommunityEvent[]> {
    return apiClient.get('/events/nearby', {
      params: { latitude, longitude, radius },
    });
  },
};
