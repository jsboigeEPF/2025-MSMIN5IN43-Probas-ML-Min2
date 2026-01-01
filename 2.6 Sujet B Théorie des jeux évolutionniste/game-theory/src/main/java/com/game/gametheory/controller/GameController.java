package com.game.gametheory.controller;

import com.game.gametheory.engine.*;
import com.game.gametheory.model.*;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/game")
public class GameController {
  private GameEngine engine;

  /**
   * Démarre une nouvelle partie.
   *
   */
  @PostMapping("/start")
  public void start(
          @RequestParam int hawks,
          @RequestParam int doves,
          @RequestParam(defaultValue = "0") int grudges,
          @RequestParam(defaultValue = "0") int detectives){
    engine = new GameEngine(hawks, doves, grudges, detectives);
  }

  /**
   * Avance la simulation d'un jour.
   *
   * @return snapshot de l'état du jeu
   */
  @GetMapping("/step")
  public GameSnapshot step() {
    if (engine == null) {
      throw new IllegalStateException("La partie n'a pas été démarrée. Utilisez /start.");
    }
    return engine.nextDay();
  }
}