import { createClient } from '@supabase/supabase-js';
import { projectId, publicAnonKey } from './supabase/info';
import { SignupData, SigninData, AuthResponse, Profile } from './types';

const supabase = createClient(
  `https://${projectId}.supabase.co`,
  publicAnonKey
);

class AuthService {
  private currentUser: any = null;
  private currentProfile: Profile | null = null;
  private authListeners: ((user: any, profile: Profile | null) => void)[] = [];

  constructor() {
    this.initializeAuth();
  }

  private async initializeAuth() {
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (session?.user) {
        this.currentUser = session.user;
        await this.loadProfile(session.user.id);
      }
    } catch (error) {
      console.error('Auth initialization failed:', error);
    }
  }

  private async loadProfile(userId: string) {
    try {
      const response = await fetch(`https://${projectId}.supabase.co/functions/v1/make-server-ec56cf0b/profile/${userId}`, {
        headers: {
          'Authorization': `Bearer ${publicAnonKey}`,
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const data = await response.json();
        this.currentProfile = data.profile;
      }
    } catch (error) {
      console.error('Failed to load profile:', error);
    }
  }

  onAuthStateChange(callback: (user: any, profile: Profile | null) => void) {
    this.authListeners.push(callback);
    // Call immediately with current state
    callback(this.currentUser, this.currentProfile);
    
    return () => {
      const index = this.authListeners.indexOf(callback);
      if (index > -1) {
        this.authListeners.splice(index, 1);
      }
    };
  }

  private notifyAuthListeners() {
    this.authListeners.forEach(callback => {
      callback(this.currentUser, this.currentProfile);
    });
  }

  async signup(data: SignupData): Promise<AuthResponse> {
    try {
      const response = await fetch(`https://${projectId}.supabase.co/functions/v1/make-server-ec56cf0b/auth/signup`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${publicAnonKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      const result = await response.json();

      if (result.success) {
        this.currentUser = result.user;
        this.currentProfile = result.profile;
        this.notifyAuthListeners();
      }

      return result;
    } catch (error) {
      console.error('Signup failed:', error);
      return { success: false, error: 'Signup failed' };
    }
  }

  async signin(data: SigninData): Promise<AuthResponse> {
    try {
      const { data: authData, error } = await supabase.auth.signInWithPassword(data);

      if (error) {
        return { success: false, error: error.message };
      }

      this.currentUser = authData.user;
      await this.loadProfile(authData.user.id);
      this.notifyAuthListeners();

      return { 
        success: true, 
        user: authData.user, 
        profile: this.currentProfile,
        session: authData.session 
      };
    } catch (error) {
      console.error('Signin failed:', error);
      return { success: false, error: 'Signin failed' };
    }
  }

  async signout(): Promise<{ success: boolean; error?: string }> {
    try {
      const { error } = await supabase.auth.signOut();

      if (error) {
        return { success: false, error: error.message };
      }

      this.currentUser = null;
      this.currentProfile = null;
      this.notifyAuthListeners();

      return { success: true };
    } catch (error) {
      console.error('Signout failed:', error);
      return { success: false, error: 'Signout failed' };
    }
  }

  getCurrentUser() {
    return this.currentUser;
  }

  getCurrentProfile() {
    return this.currentProfile;
  }

  getAccessToken(): string | null {
    // For demo purposes, we'll use a mock token
    // In a real app, this would get the actual session token
    return this.currentUser ? 'mock-access-token' : null;
  }

  isAuthenticated(): boolean {
    return this.currentUser !== null;
  }

  async initializeDemoData() {
    try {
      const response = await fetch(`https://${projectId}.supabase.co/functions/v1/make-server-ec56cf0b/init-demo`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${publicAnonKey}`,
          'Content-Type': 'application/json',
        },
      });

      return await response.json();
    } catch (error) {
      console.error('Failed to initialize demo data:', error);
      return { success: false, error: 'Failed to initialize demo data' };
    }
  }
}

export const authService = new AuthService();