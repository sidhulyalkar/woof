'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { X, Loader2, ArrowLeft, Link as LinkIcon, ImagePlus } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { useCreatePost } from '@/lib/api/hooks';
import { useSessionStore } from '@/store/session';
import { useUIStore } from '@/store/ui';
import { EmojiAvatar, getEmojiForId, getVariantForId } from '@/components/ui/EmojiAvatar';

export default function CreatePostPage() {
  const router = useRouter();
  const { user, pets } = useSessionStore();
  const { showToast } = useUIStore();
  const [content, setContent] = useState('');
  const [images, setImages] = useState<string[]>([]);
  const [imageUrlInput, setImageUrlInput] = useState('');
  const [selectedPet, setSelectedPet] = useState<string | null>(pets[0]?.id || null);
  const [showImageInput, setShowImageInput] = useState(false);

  const createPostMutation = useCreatePost({
    onSuccess: () => {
      showToast({ message: 'Post created successfully!', type: 'success' });
      router.push('/');
    },
    onError: (error) => {
      showToast({ message: error.message || 'Failed to create post', type: 'error' });
    },
  });

  const handleAddImageUrl = () => {
    if (imageUrlInput.trim()) {
      setImages((prev) => [...prev, imageUrlInput.trim()]);
      setImageUrlInput('');
      setShowImageInput(false);
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

  return (
    <div className="min-h-screen bg-gradient-to-b from-background via-background to-muted/5">
      {/* Modern Header - Minimal & Clean */}
      <div className="sticky top-0 z-50 backdrop-blur-xl bg-background/80 border-b border-border/5">
        <div className="max-w-2xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={() => router.back()}
              className="w-9 h-9 flex items-center justify-center rounded-full hover:bg-muted/50 transition-all duration-200 active:scale-95"
            >
              <ArrowLeft className="h-5 w-5" />
            </button>
            <h1 className="text-lg font-semibold">New Post</h1>
          </div>
          <Button
            onClick={handleSubmit}
            disabled={createPostMutation.isPending || (!content.trim() && images.length === 0)}
            size="sm"
            className="h-9 px-6 rounded-full font-medium bg-gradient-to-r from-violet-500 to-purple-500 hover:from-violet-600 hover:to-purple-600 shadow-sm hover:shadow-md transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {createPostMutation.isPending ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Posting
              </>
            ) : (
              'Post'
            )}
          </Button>
        </div>
      </div>

      {/* Content Area */}
      <div className="max-w-2xl mx-auto px-4 py-6 space-y-4">
        {/* User Info */}
        <div className="flex items-start gap-3">
          <EmojiAvatar
            emoji={getEmojiForId(user?.id || '', 'user')}
            variant={getVariantForId(user?.id || '')}
            size="lg"
          />
          <div className="flex-1 space-y-1">
            <p className="font-semibold text-[15px]">{user?.username}</p>
            {pets.length > 0 && (
              <select
                value={selectedPet || ''}
                onChange={(e) => setSelectedPet(e.target.value || null)}
                className="text-sm bg-muted/30 border border-border/30 rounded-lg px-2 py-1 outline-none hover:bg-muted/50 focus:ring-2 focus:ring-violet-500/20 transition-all cursor-pointer"
              >
                <option value="">Posting alone</option>
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
          placeholder="What's happening?"
          className="min-h-[200px] resize-none bg-transparent border-none text-[15px] leading-relaxed focus-visible:ring-0 focus-visible:ring-offset-0 placeholder:text-muted-foreground/50 px-0 text-foreground/90"
          maxLength={500}
          autoFocus
        />

        {/* Character Count */}
        <div className="flex justify-end">
          <span className={`text-xs font-medium transition-colors ${
            content.length > 450 ? 'text-destructive' : 'text-muted-foreground/60'
          }`}>
            {content.length}/500
          </span>
        </div>

        {/* Image Preview Grid */}
        {images.length > 0 && (
          <div className="grid grid-cols-2 gap-2 animate-in fade-in duration-300">
            {images.map((url, index) => (
              <div
                key={index}
                className="relative aspect-square rounded-xl overflow-hidden bg-muted/20 border border-border/10 group"
              >
                <img
                  src={url}
                  alt={`Image ${index + 1}`}
                  className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                  onError={(e) => {
                    e.currentTarget.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="100" height="100"%3E%3Crect fill="%23f5f5f5" width="100" height="100"/%3E%3Ctext fill="%23999" x="50%25" y="50%25" text-anchor="middle" dy=".3em" font-family="system-ui"%3EInvalid Image%3C/text%3E%3C/svg%3E';
                  }}
                />
                <button
                  onClick={() => handleRemoveImage(index)}
                  className="absolute top-2 right-2 w-7 h-7 flex items-center justify-center bg-black/80 backdrop-blur-sm rounded-full opacity-0 group-hover:opacity-100 hover:bg-black transition-all duration-200 hover:scale-110"
                >
                  <X className="h-4 w-4 text-white" />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Image URL Input */}
        {showImageInput ? (
          <div className="p-4 bg-muted/30 border border-border/30 rounded-xl space-y-3 animate-in slide-in-from-top-2 duration-200">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm font-medium">
                <LinkIcon className="h-4 w-4 text-violet-500" />
                <span>Add Image URL</span>
              </div>
              <button
                onClick={() => setShowImageInput(false)}
                className="w-6 h-6 flex items-center justify-center rounded-full hover:bg-muted/50 transition-colors"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="flex gap-2">
              <Input
                value={imageUrlInput}
                onChange={(e) => setImageUrlInput(e.target.value)}
                placeholder="https://example.com/image.jpg"
                className="flex-1 h-10 rounded-lg bg-background border-border/50 focus-visible:ring-violet-500/30"
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    handleAddImageUrl();
                  }
                }}
              />
              <Button
                onClick={handleAddImageUrl}
                disabled={!imageUrlInput.trim()}
                variant="default"
                size="sm"
                className="h-10 px-4 rounded-lg bg-violet-500 hover:bg-violet-600"
              >
                Add
              </Button>
            </div>
          </div>
        ) : (
          <button
            onClick={() => setShowImageInput(true)}
            className="w-full p-3 flex items-center justify-center gap-2 bg-muted/30 hover:bg-muted/50 border border-border/30 border-dashed rounded-xl transition-all duration-200 group"
          >
            <ImagePlus className="h-5 w-5 text-muted-foreground group-hover:text-violet-500 transition-colors" />
            <span className="text-sm font-medium text-muted-foreground group-hover:text-foreground transition-colors">
              Add image URL
            </span>
          </button>
        )}

        {/* Bottom Spacing */}
        <div className="h-20" />
      </div>
    </div>
  );
}
