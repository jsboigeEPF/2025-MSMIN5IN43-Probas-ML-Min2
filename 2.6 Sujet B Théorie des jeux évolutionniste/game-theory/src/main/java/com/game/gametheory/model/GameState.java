package com.game.gametheory.model;

public class GameState {

    private int day;
    private int hawks;
    private int doves;
    private int grudges;
    private int detectives;

    public GameState(int day, int hawks, int doves, int grudges, int detectives) {
        this.day = day;
        this.hawks = hawks;
        this.doves = doves;
        this.grudges = grudges;
        this.detectives = detectives;
    }

    public int getDay() {
        return day;
    }

    public int getHawks() {
        return hawks;
    }

    public int getDoves() {
        return doves;
    }

    public int getGrudges() {
        return grudges;
    }

    public int getDetectives() {
        return detectives;
    }
}