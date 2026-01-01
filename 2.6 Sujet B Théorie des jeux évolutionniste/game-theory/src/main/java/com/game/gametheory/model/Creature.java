package com.game.gametheory.model;

import java.util.HashSet;
import java.util.Set;

public abstract class Creature {
  protected double food = 0;
  public Position position;

  public Creature(Position p) {
    position = p;
  }

  public double getFood() {
    return food;
  }

  public void resetFood() {
    food = 0;
  }

  public void addFood(double f) {
    food += f;
  }

  public boolean survives() {
    if (food >= 1) return true;
    if (food == 0.5) return Math.random() < 0.5;
    return false;
  }

  public boolean reproduces() {
    if (food >= 2) return true;
    if (food == 1.5) return Math.random() < 0.5;
    return false;
  }

  public abstract Species getSpecies();
}