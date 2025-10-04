'use client';

import { useState, useRef } from 'react';
import { X, Image as ImageIcon, MapPin, Activity, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Avatar, AvatarImage, AvatarFallback } from '@/components/ui/avatar';
import { useCreatePost, useUploadImage } from '@/lib/api/hooks';
import { useSessionStore } from '@/store/session';
import { useUIStore } from '@/store/ui';

interface CreatePostModalProps {
  onClose: () => void;
}

export function CreatePostModal({ onClose }: CreatePostModalProps) {
  const { user, pets } = useSessionStore();
  const { showToast } = useUIStore();
  const [content, setContent] = useState('');
  const [images, setImages] = useState<string[]>([]);
  const [selectedPet, setSelectedPet] = useState<string | null>(pets[0]?.id || null);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const createPostMutation = useCreatePost({
    onSuccess: () => {
      showToast({ message: 'Post created successfully!', type: 'success' });
      onClose();
    },
    onError: (error) => {
      showToast({ message: error.message || 'Failed to create post', type: 'error' });
    },
  });

  const uploadImageMutation = useUploadImage();

  const handleImageSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    setIsUploading(true);
    try {
      const uploadPromises = files.map((file) => uploadImageMutation.mutateAsync(file));
      const results = await Promise.all(uploadPromises);
      const urls = results.map((result) => result.url);
      setImages((prev) => [...prev, ...urls]);
    } catch (error) {
      showToast({ message: 'Failed to upload images', type: 'error' });
    } finally {
      setIsUploading(false);
    }
  };

  const handleRemoveImage = (index: number) => {
    setImages((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = () => {
    if (!content.trim() && images.length === 0) {
      showToast({ message: 'Please add some content or images', type: 'warning' });
      return;
    }

    createPostMutation.mutate({
      content: content.trim(),
      images,
      petId: selectedPet || undefined,
    });
  };

  const selectedPetData = pets.find((pet) => pet.id === selectedPet);

  return (
    <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center p-0 sm:p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative w-full max-w-2xl bg-card/98 backdrop-blur-2xl border border-border/30 rounded-t-3xl sm:rounded-3xl shadow-2xl max-h-[90vh] overflow-hidden flex flex-col animate-in slide-in-from-bottom-4 sm:slide-in-from-bottom-0 sm:fade-in duration-300">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-border/10">
          <h2 className="text-lg font-bold">Create Post</h2>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="h-9 w-9 rounded-full hover:bg-muted/50 transition-colors"
          >
            <X className="h-5 w-5" />
          </Button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4">
          {/* User Info */}
          <div className="flex items-center gap-3">
            <Avatar className="h-11 w-11 ring-2 ring-primary/10">
              <AvatarImage src={user?.avatar} />
              <AvatarFallback className="bg-gradient-to-br from-purple-500 to-pink-500 text-white text-sm font-medium">
                {user?.username?.[0]?.toUpperCase() || 'U'}
              </AvatarFallback>
            </Avatar>
            <div className="flex-1">
              <p className="font-semibold text-[15px]">{user?.username}</p>
              {pets.length > 0 && (
                <select
                  value={selectedPet || ''}
                  onChange={(e) => setSelectedPet(e.target.value || null)}
                  className="text-xs bg-transparent border-none outline-none text-muted-foreground cursor-pointer hover:text-primary transition-colors"
                >
                  <option value="">No pet selected</option>
                  {pets.map((pet) => (
                    <option key={pet.id} value={pet.id}>
                      with {pet.name}
                    </option>
                  ))}
                </select>
              )}
            </div>
          </div>

          {/* Text Input */}
          <Textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="What's on your mind?"
            className="min-h-[140px] resize-none bg-transparent border-none text-[15px] leading-relaxed focus-visible:ring-0 focus-visible:ring-offset-0 placeholder:text-muted-foreground/60"
            maxLength={500}
          />

          {/* Image Preview */}
          {images.length > 0 && (
            <div className="grid grid-cols-2 gap-2">
              {images.map((url, index) => (
                <div key={index} className="relative aspect-square rounded-2xl overflow-hidden bg-muted/20">
                  <img
                    src={url}
                    alt={`Upload ${index + 1}`}
                    className="w-full h-full object-cover"
                  />
                  <button
                    onClick={() => handleRemoveImage(index)}
                    className="absolute top-2 right-2 p-1.5 bg-black/70 backdrop-blur-sm rounded-full hover:bg-black/90 transition-all duration-200 hover:scale-110"
                  >
                    <X className="h-3.5 w-3.5 text-white" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Character Count */}
          <div className="text-right text-xs text-muted-foreground/70">
            {content.length}/500
          </div>
        </div>

        {/* Actions */}
        <div className="border-t border-border/10 px-5 py-3.5">
          <div className="flex items-center justify-between mb-3">
            <div className="flex gap-1">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                multiple
                onChange={handleImageSelect}
                className="hidden"
              />
              <Button
                variant="ghost"
                size="icon"
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
                className="h-9 w-9 rounded-xl hover:bg-muted/50 transition-colors"
              >
                {isUploading ? (
                  <Loader2 className="h-[18px] w-[18px] animate-spin" />
                ) : (
                  <ImageIcon className="h-5 w-5" />
                )}
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-9 w-9 rounded-xl hover:bg-muted/50 transition-colors"
                disabled
              >
                <MapPin className="h-[18px] w-[18px]" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-9 w-9 rounded-xl hover:bg-muted/50 transition-colors"
                disabled
              >
                <Activity className="h-[18px] w-[18px]" />
              </Button>
            </div>
          </div>

          <Button
            onClick={handleSubmit}
            disabled={createPostMutation.isPending || isUploading}
            className="w-full h-11 rounded-xl font-semibold text-[15px] bg-gradient-to-r from-primary to-pink-500 hover:from-primary/90 hover:to-pink-500/90 transition-all duration-200 disabled:opacity-50"
          >
            {createPostMutation.isPending ? (
              <>
                <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                Posting...
              </>
            ) : (
              'Post'
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}
