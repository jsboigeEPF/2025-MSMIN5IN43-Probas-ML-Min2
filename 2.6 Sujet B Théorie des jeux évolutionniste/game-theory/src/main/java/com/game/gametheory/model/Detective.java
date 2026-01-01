package com.game.gametheory.model;

import java.util.HashSet;
import java.util.Set;

public class Detective extends Creature {

    // Ensemble pour mémoriser les Doves exploitables
    private Set<Creature> doveMemory = new HashSet<>();

    public Detective(Position p) {
        super(p);
    }

    /**
     * Le Detective est toujours de type DETECTIVE
     */
    @Override
    public Species getSpecies() {
        return Species.DETECTIVE;
    }

    /**
     * Mémorise une Dove après interaction
     */
    public void rememberDove(Creature other) {
        if (other.getSpecies() == Species.DOVE) {
            doveMemory.add(other);
        }
    }

    /**
     * Le Detective se comporte comme Hawk
     * uniquement face aux Doves mémorisées
     */
    public boolean behavesAsHawkAgainst(Creature other) {
        return doveMemory.contains(other) && other.getSpecies() == Species.DOVE;
    }

    /**
     * Retourne la mémoire (utile pour le moteur)
     */
    public Set<Creature> getDoveMemory() {
        return doveMemory;
    }
}