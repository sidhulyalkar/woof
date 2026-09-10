import React from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useRelationshipScope } from '../../relationship/relationship-scope';
import { colors } from '../../theme/tokens';

type Props = {
  onBeforeSelect?: () => void;
};

export function RelationshipScopeBar({ onBeforeSelect }: Props) {
  const { pets, selectedPetId, selectedPet, loading, error, selectPet } = useRelationshipScope();

  if (loading && pets.length === 0) {
    return (
      <View style={styles.loadingCard} accessibilityRole="progressbar">
        <ActivityIndicator size="small" color={colors.primary[600]} />
        <Text style={styles.loadingText}>Checking which dog you’re with…</Text>
      </View>
    );
  }

  if (!selectedPet) {
    return (
      <View style={styles.unavailableCard} accessibilityRole={error ? 'alert' : undefined}>
        <Ionicons name="paw-outline" size={18} color={colors.gray[600]} />
        <Text style={styles.unavailableText}>
          {error ?? 'No authorized dog relationship is available for this view.'}
        </Text>
      </View>
    );
  }

  if (pets.length === 1) {
    return (
      <View
        style={styles.singleCard}
        accessible
        accessibilityLabel={`Viewing your relationship with ${selectedPet.name}`}
      >
        <View style={styles.scopeIcon}>
          <Ionicons name="paw" size={17} color={colors.primary[700]} />
        </View>
        <View style={styles.copy}>
          <Text style={styles.eyebrow}>WITH</Text>
          <Text style={styles.selectedName}>{selectedPet.name}</Text>
        </View>
        <Text style={styles.singleHint}>This view is just for this relationship.</Text>
      </View>
    );
  }

  return (
    <View style={styles.wrapper}>
      <View style={styles.headingRow}>
        <View>
          <Text style={styles.eyebrow}>RELATIONSHIP</Text>
          <Text style={styles.heading}>Who are you with?</Text>
        </View>
        <Text style={styles.headingHint}>Each dog keeps a separate history.</Text>
      </View>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.chipRow}
      >
        {pets.map((pet) => {
          const selected = pet.id === selectedPetId;
          return (
            <Pressable
              key={pet.id}
              accessibilityRole="button"
              accessibilityState={{ selected }}
              accessibilityLabel={`${selected ? 'Viewing' : 'View'} relationship with ${pet.name}`}
              onPress={() => {
                if (selected) return;
                onBeforeSelect?.();
                selectPet(pet.id);
              }}
              style={[styles.chip, selected && styles.chipSelected]}
            >
              <Ionicons
                name={selected ? 'paw' : 'paw-outline'}
                size={16}
                color={selected ? colors.primary[800] : colors.gray[600]}
              />
              <Text style={[styles.chipText, selected && styles.chipTextSelected]}>{pet.name}</Text>
            </Pressable>
          );
        })}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: {
    marginTop: 16,
    paddingVertical: 14,
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: colors.gray[200],
  },
  headingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    gap: 12,
  },
  copy: { flex: 1 },
  eyebrow: { color: colors.text.secondary, fontSize: 10, fontWeight: '800', letterSpacing: 1.2 },
  heading: { marginTop: 3, color: colors.text.primary, fontSize: 15, fontWeight: '800' },
  headingHint: {
    flex: 1,
    color: colors.text.secondary,
    fontSize: 10,
    lineHeight: 15,
    textAlign: 'right',
  },
  chipRow: { gap: 8, paddingTop: 10, paddingRight: 18 },
  chip: {
    minHeight: 44,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    paddingHorizontal: 14,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: colors.gray[300],
    backgroundColor: '#ffffff',
  },
  chipSelected: {
    borderColor: colors.primary[300],
    backgroundColor: colors.primary[100],
  },
  chipText: { color: colors.gray[700], fontSize: 13, fontWeight: '700' },
  chipTextSelected: { color: colors.primary[900] },
  singleCard: {
    minHeight: 58,
    marginTop: 16,
    paddingHorizontal: 14,
    paddingVertical: 10,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    borderRadius: 16,
    backgroundColor: colors.primary[50],
    borderWidth: 1,
    borderColor: colors.primary[100],
  },
  scopeIcon: {
    width: 36,
    height: 36,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#ffffff',
  },
  selectedName: { marginTop: 1, color: colors.text.primary, fontSize: 15, fontWeight: '800' },
  singleHint: {
    maxWidth: 132,
    color: colors.text.secondary,
    fontSize: 10,
    lineHeight: 14,
    textAlign: 'right',
  },
  loadingCard: {
    minHeight: 52,
    marginTop: 16,
    paddingHorizontal: 14,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 9,
    borderRadius: 15,
    backgroundColor: colors.gray[50],
  },
  loadingText: { color: colors.text.secondary, fontSize: 12 },
  unavailableCard: {
    minHeight: 52,
    marginTop: 16,
    paddingHorizontal: 14,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 9,
    borderRadius: 15,
    backgroundColor: colors.gray[50],
    borderWidth: 1,
    borderColor: colors.gray[200],
  },
  unavailableText: { flex: 1, color: colors.text.secondary, fontSize: 12, lineHeight: 17 },
});
