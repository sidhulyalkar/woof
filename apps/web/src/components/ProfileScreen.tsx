'use client';

import { useState } from 'react';
import { Settings, MapPin, Calendar, Award, Camera, MessageCircle, Plus, Loader2 } from 'lucide-react';
import { ProfileAvatar, getPlaceholderAvatar } from '@/components/ui/ProfileAvatar';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useSessionStore } from '@/store/session';
import { useFeed } from '@/lib/api/hooks';
import { PostCard } from '@/components/feed/PostCard';
import { useUIStore } from '@/store/ui';
import { formatDistanceToNow } from 'date-fns';
import { AddPetModal } from '@/components/profile/AddPetModal';

export function ProfileScreen() {
  const { user, pets } = useSessionStore();
  const { showToast } = useUIStore();
  const [activeTab, setActiveTab] = useState('posts');
  const [showAddPetModal, setShowAddPetModal] = useState(false);

  // Get user's posts (we'll filter by user ID later)
  const { data: allPosts, isLoading } = useFeed();
  const userPosts = allPosts?.filter(post => post.userId === user?.id) || [];

  const handleEditProfile = () => {
    showToast({ message: 'Profile editing coming soon!', type: 'info' });
  };

  const handleAddPet = () => {
    setShowAddPetModal(true);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white">
      {/* Figma Glass Header: Centered, frosted glass effect */}
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-white/80 border-b border-gray-200/60 shadow-sm">
        <div className="max-w-2xl mx-auto flex items-center justify-between px-4 h-14">
          <h1 className="text-lg font-semibold text-gray-900">
            Profile
          </h1>
          <button className="w-9 h-9 flex items-center justify-center rounded-full hover:bg-gray-100/80 backdrop-blur-sm transition-all">
            <Settings className="h-5 w-5 text-gray-700" />
          </button>
        </div>
      </header>

      <div className="pb-20">
        <div className="max-w-2xl mx-auto">
          {/* Profile Header - Premium Glass Style with ProfileAvatar */}
          <div className="mt-3 mx-4 bg-white/90 backdrop-blur-xl border border-gray-200/40 rounded-3xl shadow-lg p-6">
          <div className="flex items-start gap-4">
            {/* ProfileAvatar */}
            <div className="relative">
              <ProfileAvatar
                type="user"
                src={user?.avatar || getPlaceholderAvatar(user?.username || 'User', 'user')}
                fallbackText={(user?.username || 'User').slice(0, 2).toUpperCase()}
                size="2xl"
              />
              <button className="absolute bottom-0 right-0 w-7 h-7 flex items-center justify-center bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 rounded-full shadow-md transition-all active:scale-95">
                <Camera className="h-3.5 w-3.5 text-white" />
              </button>
            </div>

            {/* User Info */}
            <div className="flex-1 min-w-0">
              <h2 className="text-lg font-semibold text-gray-900 mb-1">{user?.username || 'Anonymous'}</h2>

              <div className="flex items-center gap-3 text-xs text-gray-600 mb-2">
                {user?.location && (
                  <div className="flex items-center gap-1">
                    <MapPin className="h-3 w-3" />
                    {user.location}
                  </div>
                )}
                <div className="flex items-center gap-1">
                  <Calendar className="h-3 w-3" />
                  Joined {user?.createdAt ? formatDistanceToNow(new Date(user.createdAt), { addSuffix: true }) : 'recently'}
                </div>
              </div>

              {user?.bio && (
                <p className="text-sm text-gray-700 mb-3 line-clamp-2">
                  {user.bio}
                </p>
              )}

              {/* Stats - Inline */}
              <div className="flex items-center gap-4 text-sm mb-3">
                <div>
                  <span className="font-semibold text-gray-900">{userPosts.length}</span>
                  <span className="text-gray-600 ml-1">Posts</span>
                </div>
                <div>
                  <span className="font-semibold text-gray-900">{pets.length}</span>
                  <span className="text-gray-600 ml-1">Pets</span>
                </div>
                <div>
                  <span className="font-semibold text-gray-900">{user?.points || 0}</span>
                  <span className="text-gray-600 ml-1">Points</span>
                </div>
              </div>

              {/* Edit Button */}
              <button
                onClick={handleEditProfile}
                className="px-4 py-1.5 bg-gray-100 hover:bg-gray-200 border border-gray-200 rounded text-sm font-medium text-gray-900 transition-colors"
              >
                Edit Profile
              </button>
            </div>
          </div>
          </div>

          {/* Content Tabs - Figma Glass Style: In centered container */}
          <div className="mt-3 mx-4 bg-white/80 backdrop-blur-md border border-gray-200/60 rounded-2xl shadow-sm overflow-hidden">
            <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-3 bg-transparent h-12 p-0 gap-0 border-0">
              <TabsTrigger
                value="posts"
                className="rounded-none border-b-2 border-transparent data-[state=active]:border-gray-900 data-[state=active]:text-gray-900 text-gray-600 font-medium"
              >
                Posts
              </TabsTrigger>
              <TabsTrigger
                value="pets"
                className="rounded-none border-b-2 border-transparent data-[state=active]:border-gray-900 data-[state=active]:text-gray-900 text-gray-600 font-medium"
              >
                Pets
              </TabsTrigger>
              <TabsTrigger
                value="activity"
                className="rounded-none border-b-2 border-transparent data-[state=active]:border-gray-900 data-[state=active]:text-gray-900 text-gray-600 font-medium"
              >
                Activity
              </TabsTrigger>
            </TabsList>

            {/* Posts Tab - Figma Style: No extra spacing */}
            <TabsContent value="posts" className="mt-0">
              {isLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin text-gray-400" />
                </div>
              ) : userPosts.length > 0 ? (
                <div>
                  {userPosts.map((post) => <PostCard key={post.id} post={post} />)}
                </div>
              ) : (
                <div className="p-12 text-center border-b border-gray-100">
                  <div className="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center mx-auto mb-3">
                    <MessageCircle className="h-6 w-6 text-gray-400" />
                  </div>
                  <p className="text-sm font-medium text-gray-900 mb-1">No posts yet</p>
                  <p className="text-xs text-gray-600">Start sharing your adventures!</p>
                </div>
              )}
            </TabsContent>

            {/* Pets Tab - Figma Style: Clean cards */}
            <TabsContent value="pets" className="mt-0">
              {pets.length > 0 && (
                <div className="border-b border-gray-100">
                  {pets.map((pet, index) => (
                    <div
                      key={pet.id}
                      className={`px-4 py-4 flex items-start gap-3 ${index !== pets.length - 1 ? 'border-b border-gray-100' : ''}`}
                    >
                      <ProfileAvatar
                        type="pet"
                        src={pet.avatar || getPlaceholderAvatar(pet.name, 'pet')}
                        fallbackText={pet.name.slice(0, 2).toUpperCase()}
                        size="lg"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <h3 className="text-base font-semibold text-gray-900">{pet.name}</h3>
                          <span className="px-2 py-0.5 bg-gray-100 text-gray-700 text-xs rounded-full font-medium">
                            {pet.species}
                          </span>
                        </div>
                        {pet.breed && (
                          <p className="text-sm text-gray-600 mb-1">{pet.breed}</p>
                        )}
                        {pet.bio && (
                          <p className="text-sm text-gray-700 line-clamp-2">{pet.bio}</p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <button
                onClick={handleAddPet}
                className="w-full px-4 py-4 border-b border-gray-100 flex items-center justify-center gap-2 text-gray-600 hover:bg-gray-50 transition-colors"
              >
                <Plus className="h-5 w-5" />
                <span className="font-medium text-sm">Add New Pet</span>
              </button>
            </TabsContent>

            {/* Activity Tab - Figma Style: Simple stats */}
            <TabsContent value="activity" className="mt-0">
              <div className="px-4 py-6 border-b border-gray-100">
                <h3 className="text-sm font-semibold text-gray-900 mb-4">This Month</h3>
                <div className="grid grid-cols-2 gap-3">
                  <div className="border border-gray-200 rounded p-3 text-center">
                    <p className="text-xl font-semibold text-gray-900">0 km</p>
                    <p className="text-xs text-gray-600 mt-0.5">Distance</p>
                  </div>
                  <div className="border border-gray-200 rounded p-3 text-center">
                    <p className="text-xl font-semibold text-gray-900">0</p>
                    <p className="text-xs text-gray-600 mt-0.5">Calories</p>
                  </div>
                  <div className="border border-gray-200 rounded p-3 text-center">
                    <p className="text-xl font-semibold text-gray-900">0h</p>
                    <p className="text-xs text-gray-600 mt-0.5">Active Time</p>
                  </div>
                  <div className="border border-gray-200 rounded p-3 text-center">
                    <p className="text-xl font-semibold text-gray-900">0</p>
                    <p className="text-xs text-gray-600 mt-0.5">Activities</p>
                  </div>
                </div>
              </div>

              <div className="px-4 py-6 border-b border-gray-100">
                <h3 className="text-sm font-semibold text-gray-900 mb-4">Achievements</h3>
                <div className="text-center py-6">
                  <div className="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center mx-auto mb-3">
                    <Award className="h-6 w-6 text-gray-400" />
                  </div>
                  <p className="text-xs text-gray-600">No achievements yet</p>
                  <p className="text-xs text-gray-500 mt-1">Start tracking activities to earn badges!</p>
                </div>
              </div>
            </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>

      {/* Add Pet Modal */}
      <AddPetModal isOpen={showAddPetModal} onClose={() => setShowAddPetModal(false)} />
    </div>
  );
}
