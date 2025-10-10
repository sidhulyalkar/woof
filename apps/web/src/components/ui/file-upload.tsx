'use client';

import { useState, useRef, ChangeEvent } from 'react';
import { Upload, X, Image as ImageIcon, Video } from 'lucide-react';
import { Button } from './button';
import { cn } from '@/lib/utils';
import { validateFileUpload } from '@/lib/security';

interface FileUploadProps {
  onUpload: (files: File[]) => void | Promise<void>;
  onRemove?: (index: number) => void;
  multiple?: boolean;
  accept?: string;
  maxSizeMB?: number;
  preview?: boolean;
  className?: string;
  disabled?: boolean;
  value?: File[];
}

export function FileUpload({
  onUpload,
  onRemove,
  multiple = false,
  accept = 'image/*,video/*',
  maxSizeMB = 10,
  preview = true,
  className,
  disabled = false,
  value = [],
}: FileUploadProps) {
  const [files, setFiles] = useState<File[]>(value);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    setError(null);

    if (selectedFiles.length === 0) return;

    // Validate files
    const allowedTypes = accept
      .split(',')
      .map((type) => type.trim())
      .filter(Boolean);

    for (const file of selectedFiles) {
      const validation = validateFileUpload(file, {
        maxSizeBytes: maxSizeMB * 1024 * 1024,
        allowedTypes:
          allowedTypes.length > 0
            ? allowedTypes.map((type) =>
                type.replace('/*', '').includes('/')
                  ? type
                  : `${type.replace('*', 'jpeg')}`,
              )
            : undefined,
      });

      if (!validation.valid) {
        setError(validation.error || 'Invalid file');
        return;
      }
    }

    const newFiles = multiple ? [...files, ...selectedFiles] : selectedFiles;
    setFiles(newFiles);

    // Call onUpload
    try {
      setUploading(true);
      await onUpload(newFiles);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }

    // Reset input
    if (inputRef.current) {
      inputRef.current.value = '';
    }
  };

  const handleRemove = (index: number) => {
    const newFiles = files.filter((_, i) => i !== index);
    setFiles(newFiles);
    onRemove?.(index);
  };

  const handleClick = () => {
    inputRef.current?.click();
  };

  const getFileIcon = (file: File) => {
    if (file.type.startsWith('video/')) {
      return <Video className="h-8 w-8" />;
    }
    return <ImageIcon className="h-8 w-8" />;
  };

  const getPreviewUrl = (file: File): string => {
    return URL.createObjectURL(file);
  };

  return (
    <div className={cn('space-y-4', className)}>
      {/* Upload Button */}
      <Button
        type="button"
        variant="outline"
        onClick={handleClick}
        disabled={disabled || uploading}
        className="w-full"
      >
        <Upload className="mr-2 h-4 w-4" />
        {uploading
          ? 'Uploading...'
          : multiple
            ? 'Upload Files'
            : 'Upload File'}
      </Button>

      {/* Hidden File Input */}
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        onChange={handleFileChange}
        className="hidden"
        disabled={disabled || uploading}
      />

      {/* Error Message */}
      {error && (
        <div className="rounded-lg border border-destructive bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* File Previews */}
      {preview && files.length > 0 && (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4">
          {files.map((file, index) => (
            <div
              key={index}
              className="group relative aspect-square overflow-hidden rounded-lg border border-border bg-muted"
            >
              {file.type.startsWith('image/') ? (
                <img
                  src={getPreviewUrl(file)}
                  alt={file.name}
                  className="h-full w-full object-cover"
                />
              ) : (
                <div className="flex h-full w-full items-center justify-center">
                  {getFileIcon(file)}
                </div>
              )}

              {/* Remove Button */}
              <button
                type="button"
                onClick={() => handleRemove(index)}
                disabled={disabled || uploading}
                className="absolute right-1 top-1 rounded-full bg-background/80 p-1 opacity-0 transition-opacity group-hover:opacity-100"
              >
                <X className="h-4 w-4" />
              </button>

              {/* File Name */}
              <div className="absolute bottom-0 left-0 right-0 bg-background/80 p-2 text-xs truncate">
                {file.name}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
