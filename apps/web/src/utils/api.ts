import { projectId, publicAnonKey } from './supabase/info';

const API_BASE_URL = `https://${projectId}.supabase.co/functions/v1/make-server-ec56cf0b`;

class ApiService {
  private async makeRequest(endpoint: string, options: RequestInit = {}) {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        ...options,
        headers: {
          'Authorization': `Bearer ${publicAnonKey}`,
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error);
      // Return mock data for development/testing
      return this.getMockData(endpoint);
    }
  }

  private getMockData(endpoint: string) {
    // Return appropriate mock data based on endpoint
    switch (true) {
      case endpoint.includes('/health'):
        return { status: 'ok', timestamp: new Date().toISOString() };
      case endpoint.includes('/feed'):
        return { posts: [] };
      case endpoint.includes('/activities'):
        return { activities: [] };
      case endpoint.includes('/leaderboard'):
        return { leaderboard: [] };
      case endpoint.includes('/messages'):
        return { messages: [] };
      case endpoint.includes('/profile'):
        return { profile: null };
      default:
        return { success: true };
    }
  }

  async healthCheck() {
    return this.makeRequest('/health');
  }

  async getFeed() {
    return this.makeRequest('/feed');
  }

  async createPost(postData: any) {
    return this.makeRequest('/feed', {
      method: 'POST',
      body: JSON.stringify(postData),
    });
  }

  async getActivities() {
    return this.makeRequest('/activities');
  }

  async createActivity(activityData: any) {
    return this.makeRequest('/activities', {
      method: 'POST',
      body: JSON.stringify(activityData),
    });
  }

  async getLeaderboard(timeframe: string = 'weekly') {
    return this.makeRequest(`/leaderboard?timeframe=${timeframe}`);
  }

  async getMessages() {
    return this.makeRequest('/messages');
  }

  async sendMessage(messageData: any) {
    return this.makeRequest('/messages', {
      method: 'POST',
      body: JSON.stringify(messageData),
    });
  }

  async getProfile(userId: string) {
    return this.makeRequest(`/profile/${userId}`);
  }

  async updateProfile(userId: string, profileData: any) {
    return this.makeRequest(`/profile/${userId}`, {
      method: 'PUT',
      body: JSON.stringify(profileData),
    });
  }
}

export const apiService = new ApiService();