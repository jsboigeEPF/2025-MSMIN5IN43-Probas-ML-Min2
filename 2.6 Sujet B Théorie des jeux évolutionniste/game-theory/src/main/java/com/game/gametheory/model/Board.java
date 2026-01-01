package com.game.gametheory.model;
import java.util.*;
public class Board {
  public double radius;
  public List<Creature> creatures=new ArrayList<>();
  public Board(double r){radius=r;}
  public Position randomPosition(){
    double a=Math.random()*2*Math.PI;
    double d=Math.sqrt(Math.random())*radius;
    return new Position(d*Math.cos(a),d*Math.sin(a));
  }
}
