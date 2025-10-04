import React, { useState } from 'react';
import { Dog, Heart, Trophy, MessageCircle, Eye, EyeOff } from 'lucide-react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { authService } from '../utils/auth';
import { SignupData, SigninData } from '../utils/types';

interface AuthScreenProps {
  onAuthSuccess: () => void;
}

const petTypes = [
  'Dog', 'Cat', 'Bird', 'Rabbit', 'Hamster', 'Guinea Pig', 
  'Fish', 'Reptile', 'Horse', 'Other'
];

export function AuthScreen({ onAuthSuccess }: AuthScreenProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  
  // Signin form state
  const [signinData, setSigninData] = useState<SigninData>({
    email: '',
    password: ''
  });

  // Signup form state
  const [signupData, setSignupData] = useState<SignupData>({
    email: '',
    password: '',
    name: '',
    petName: '',
    petType: ''
  });

  const handleSignin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      const result = await authService.signin(signinData);
      
      if (result.success) {
        onAuthSuccess();
      } else {
        setError(result.error || 'Signin failed');
      }
    } catch (error) {
      setError('Network error. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    if (!signupData.name || !signupData.petName || !signupData.petType) {
      setError('Please fill in all fields');
      setIsLoading(false);
      return;
    }

    try {
      const result = await authService.signup(signupData);
      
      if (result.success) {
        onAuthSuccess();
      } else {
        setError(result.error || 'Signup failed');
      }
    } catch (error) {
      setError('Network error. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDemoAccess = async () => {
    setIsLoading(true);
    setError('');

    try {
      // Initialize demo data first
      await authService.initializeDemoData();
      onAuthSuccess();
    } catch (error) {
      setError('Failed to load demo. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center p-4">
      {/* Hero Section */}
      <div className="text-center mb-8 max-w-md">
        <div className="flex items-center justify-center gap-2 mb-4">
          <div className="p-3 rounded-full bg-accent/20">
            <Dog className="w-8 h-8 text-accent" />
          </div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-accent to-blue-400 bg-clip-text text-transparent">
            PetPath
          </h1>
        </div>
        
        <p className="text-muted-foreground mb-6">
          The social fitness platform for pets and their humans. Track activities, share moments, and compete with friends!
        </p>

        {/* Feature highlights */}
        <div className="flex justify-center gap-6 mb-8">
          <div className="flex flex-col items-center gap-1">
            <Heart className="w-5 h-5 text-accent" />
            <span className="text-xs text-muted-foreground">Share</span>
          </div>
          <div className="flex flex-col items-center gap-1">
            <Trophy className="w-5 h-5 text-accent" />
            <span className="text-xs text-muted-foreground">Compete</span>
          </div>
          <div className="flex flex-col items-center gap-1">
            <MessageCircle className="w-5 h-5 text-accent" />
            <span className="text-xs text-muted-foreground">Connect</span>
          </div>
        </div>
      </div>

      {/* Auth Forms */}
      <Card className="w-full max-w-md glass-card border-border/20">
        <CardHeader className="text-center">
          <CardTitle>Welcome to PetPath</CardTitle>
          <CardDescription>
            Join the community of pet lovers
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          <Tabs defaultValue="signin" className="w-full">
            <TabsList className="grid w-full grid-cols-2 mb-6 bg-surface-elevated/50">
              <TabsTrigger value="signin">Sign In</TabsTrigger>
              <TabsTrigger value="signup">Sign Up</TabsTrigger>
            </TabsList>

            {error && (
              <div className="mb-4 p-3 rounded-lg bg-destructive/20 border border-destructive/30 text-destructive text-sm">
                {error}
              </div>
            )}

            <TabsContent value="signin">
              <form onSubmit={handleSignin} className="space-y-4">
                <div>
                  <Label htmlFor="signin-email">Email</Label>
                  <Input
                    id="signin-email"
                    type="email"
                    placeholder="your@email.com"
                    value={signinData.email}
                    onChange={(e) => setSigninData(prev => ({ ...prev, email: e.target.value }))}
                    required
                    className="bg-input-background border-border/30"
                  />
                </div>
                
                <div>
                  <Label htmlFor="signin-password">Password</Label>
                  <div className="relative">
                    <Input
                      id="signin-password"
                      type={showPassword ? 'text' : 'password'}
                      placeholder="••••••••"
                      value={signinData.password}
                      onChange={(e) => setSigninData(prev => ({ ...prev, password: e.target.value }))}
                      required
                      className="bg-input-background border-border/30 pr-10"
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="absolute right-2 top-1/2 -translate-y-1/2 h-auto p-1"
                      onClick={() => setShowPassword(!showPassword)}
                    >
                      {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                    </Button>
                  </div>
                </div>

                <Button 
                  type="submit" 
                  className="w-full bg-accent hover:bg-accent/90"
                  disabled={isLoading}
                >
                  {isLoading ? 'Signing in...' : 'Sign In'}
                </Button>
              </form>
            </TabsContent>

            <TabsContent value="signup">
              <form onSubmit={handleSignup} className="space-y-4">
                <div>
                  <Label htmlFor="signup-name">Your Name</Label>
                  <Input
                    id="signup-name"
                    type="text"
                    placeholder="John Doe"
                    value={signupData.name}
                    onChange={(e) => setSignupData(prev => ({ ...prev, name: e.target.value }))}
                    required
                    className="bg-input-background border-border/30"
                  />
                </div>

                <div>
                  <Label htmlFor="signup-email">Email</Label>
                  <Input
                    id="signup-email"
                    type="email"
                    placeholder="your@email.com"
                    value={signupData.email}
                    onChange={(e) => setSignupData(prev => ({ ...prev, email: e.target.value }))}
                    required
                    className="bg-input-background border-border/30"
                  />
                </div>
                
                <div>
                  <Label htmlFor="signup-password">Password</Label>
                  <div className="relative">
                    <Input
                      id="signup-password"
                      type={showPassword ? 'text' : 'password'}
                      placeholder="••••••••"
                      value={signupData.password}
                      onChange={(e) => setSignupData(prev => ({ ...prev, password: e.target.value }))}
                      required
                      className="bg-input-background border-border/30 pr-10"
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="absolute right-2 top-1/2 -translate-y-1/2 h-auto p-1"
                      onClick={() => setShowPassword(!showPassword)}
                    >
                      {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                    </Button>
                  </div>
                </div>

                <div>
                  <Label htmlFor="pet-name">Pet's Name</Label>
                  <Input
                    id="pet-name"
                    type="text"
                    placeholder="Buddy"
                    value={signupData.petName}
                    onChange={(e) => setSignupData(prev => ({ ...prev, petName: e.target.value }))}
                    required
                    className="bg-input-background border-border/30"
                  />
                </div>

                <div>
                  <Label htmlFor="pet-type">Pet Type</Label>
                  <Select 
                    value={signupData.petType} 
                    onValueChange={(value) => setSignupData(prev => ({ ...prev, petType: value }))}
                  >
                    <SelectTrigger className="bg-input-background border-border/30">
                      <SelectValue placeholder="Select pet type" />
                    </SelectTrigger>
                    <SelectContent>
                      {petTypes.map(type => (
                        <SelectItem key={type} value={type}>{type}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <Button 
                  type="submit" 
                  className="w-full bg-accent hover:bg-accent/90"
                  disabled={isLoading}
                >
                  {isLoading ? 'Creating account...' : 'Create Account'}
                </Button>
              </form>
            </TabsContent>
          </Tabs>

          <div className="mt-6 text-center">
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-border/30"></div>
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-background px-2 text-muted-foreground">Or</span>
              </div>
            </div>
            
            <Button
              variant="outline"
              className="w-full mt-4 border-border/30 hover:bg-surface-elevated/50"
              onClick={handleDemoAccess}
              disabled={isLoading}
            >
              {isLoading ? 'Loading...' : 'Try Demo'}
            </Button>
          </div>
        </CardContent>
      </Card>

      <p className="text-xs text-muted-foreground mt-6 text-center max-w-md">
        By signing up, you agree to our Terms of Service and Privacy Policy.
        PetPath is designed to help you and your pet stay active and connected.
      </p>
    </div>
  );
}