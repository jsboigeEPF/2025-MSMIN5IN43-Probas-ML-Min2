package com.game.gametheory.model;

import java.util.HashSet;
import java.util.Set;

public class Grudge extends Creature {

    // Ensemble pour mémoriser les Hawks qui l'ont arnaquée
    private Set<Creature> hawksMemory = new HashSet<>();

    public Grudge(Position p) {
        super(p);
    }

    /**
     * Ajouter un Hawk à la mémoire si celui-ci arnaque la Grudge
     */
    public void rememberHawk(Creature hawk) {
        if (hawk.getSpecies() == Species.HAWK) {
            hawksMemory.add(hawk);
        }
    }

    /**
     * Vérifie si la Grudge doit se comporter comme Hawk face à ce Hawk
     */
    public boolean isHawkAgainst(Creature other) {
        return hawksMemory.contains(other) && other.getSpecies() == Species.HAWK;
    }

    /**
     * Retourne le type réel de la Grudge :
     * - Dove par défaut
     * - Hawk face aux Hawks mémorisés
     */
    public Species getSpecies() {
        return Species.GRUDGE; // Toujours GRUDGE comme espèce
    }

    /**
     * Méthode utilitaire pour le comportement alimentaire
     */
    public boolean behavesAsHawkAgainst(Creature other) {
        return isHawkAgainst(other);
    }

    /**
     * Retourne la mémoire (utile pour le moteur)
     */
    public Set<Creature> getHawksMemory() {
        return hawksMemory;
    }
}