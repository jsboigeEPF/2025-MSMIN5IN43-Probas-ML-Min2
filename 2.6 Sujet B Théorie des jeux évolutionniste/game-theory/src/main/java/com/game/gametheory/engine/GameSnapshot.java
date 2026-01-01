package com.game.gametheory.engine;

import com.game.gametheory.model.*;
import java.util.*;

public class GameSnapshot {
  public int day, hawks, doves, grudges, detectives;
  public List<CreatureDTO> creatures = new ArrayList<>();

  public static GameSnapshot from(List<Creature> cs, int day) {
    GameSnapshot g = new GameSnapshot();
    g.day = day;
    for (Creature c : cs) {
      CreatureDTO d = new CreatureDTO();
      d.x = c.position.x;
      d.y = c.position.y;
      d.type = c.getSpecies().name();
      g.creatures.add(d);

      if (c.getSpecies() == Species.HAWK) g.hawks++;
      else if (c.getSpecies() == Species.DOVE) g.doves++;
      else if (c.getSpecies() == Species.GRUDGE) g.grudges++;
      else if (c.getSpecies() == Species.DETECTIVE) g.detectives++;
    }
    return g;
  }
}