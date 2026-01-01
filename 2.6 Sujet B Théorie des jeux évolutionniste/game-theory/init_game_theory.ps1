$BASE = "src/main/java/com/game/gametheory"
$MODEL = "$BASE/model"
$ENGINE = "$BASE/engine"
$CTRL = "$BASE/controller"
$STATIC = "src/main/resources/static"

Write-Host "Création des dossiers..."
New-Item -ItemType Directory -Force -Path `
  $MODEL, $ENGINE, $CTRL, $STATIC | Out-Null

# ===================== pom.xml =====================
@"
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>

  <parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
    <version>3.2.1</version>
  </parent>

  <groupId>com.game</groupId>
  <artifactId>game-theory</artifactId>
  <version>0.0.1-SNAPSHOT</version>

  <properties>
    <java.version>17</java.version>
  </properties>

  <dependencies>
    <dependency>
      <groupId>org.springframework.boot</groupId>
      <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
  </dependencies>

  <build>
    <plugins>
      <plugin>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-maven-plugin</artifactId>
      </plugin>
    </plugins>
  </build>
</project>
"@ | Set-Content "pom.xml"

# ===================== Application =====================
@"
package com.game.gametheory;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class GameTheoryApplication {
  public static void main(String[] args) {
    SpringApplication.run(GameTheoryApplication.class, args);
  }
}
"@ | Set-Content "$BASE/GameTheoryApplication.java"

# ===================== MODEL =====================
@"
package com.game.gametheory.model;
public enum Species { HAWK, DOVE }
"@ | Set-Content "$MODEL/Species.java"

@"
package com.game.gametheory.model;
public class Position {
  public double x,y;
  public Position(double x,double y){this.x=x;this.y=y;}
}
"@ | Set-Content "$MODEL/Position.java"

@"
package com.game.gametheory.model;
public abstract class Creature {
  protected double food=0;
  public Position position;
  public Creature(Position p){position=p;}
  public void resetFood(){food=0;}
  public void addFood(double f){food+=f;}
  public boolean survives(){
    if(food>=1)return true;
    if(food==0.5)return Math.random()<0.5;
    return false;
  }
  public boolean reproduces(){
    if(food>=2)return true;
    if(food==1.5)return Math.random()<0.5;
    return false;
  }
  public abstract Species getSpecies();
}
"@ | Set-Content "$MODEL/Creature.java"

@"
package com.game.gametheory.model;
public class Hawk extends Creature {
  public Hawk(Position p){super(p);}
  public Species getSpecies(){return Species.HAWK;}
}
"@ | Set-Content "$MODEL/Hawk.java"

@"
package com.game.gametheory.model;
public class Dove extends Creature {
  public Dove(Position p){super(p);}
  public Species getSpecies(){return Species.DOVE;}
}
"@ | Set-Content "$MODEL/Dove.java"

@"
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
"@ | Set-Content "$MODEL/Board.java"

# ===================== ENGINE =====================
@"
package com.game.gametheory.engine;

import com.game.gametheory.model.*;
import java.util.*;

public class GameEngine {
  public int day=0;
  public Board board;

  public GameEngine(int h,int d){
    board=new Board(280);
    for(int i=0;i<h;i++)board.creatures.add(new Hawk(board.randomPosition()));
    for(int i=0;i<d;i++)board.creatures.add(new Dove(board.randomPosition()));
  }

  public GameSnapshot nextDay(){
    day++;
    List<Creature> next=new ArrayList<>();
    for(Creature c:board.creatures){
      c.resetFood();
      double r=Math.random();
      if(r<0.33)c.addFood(2);
      else if(r<0.66)c.addFood(1);
      else c.addFood(0.5);
      if(c.survives()){
        next.add(c);
        if(c.reproduces())
          next.add(c.getSpecies()==Species.HAWK?
            new Hawk(board.randomPosition()):
            new Dove(board.randomPosition()));
      }
    }
    board.creatures=next;
    return GameSnapshot.from(board.creatures,day);
  }
}
"@ | Set-Content "$ENGINE/GameEngine.java"

@"
package com.game.gametheory.engine;
public class CreatureDTO {
  public double x,y;
  public String type;
}
"@ | Set-Content "$ENGINE/CreatureDTO.java"

@"
package com.game.gametheory.engine;
import com.game.gametheory.model.*;
import java.util.*;
public class GameSnapshot {
  public int day,hawks,doves;
  public List<CreatureDTO> creatures=new ArrayList<>();
  public static GameSnapshot from(List<Creature> cs,int day){
    GameSnapshot g=new GameSnapshot(); g.day=day;
    for(Creature c:cs){
      CreatureDTO d=new CreatureDTO();
      d.x=c.position.x; d.y=c.position.y;
      d.type=c.getSpecies().name();
      g.creatures.add(d);
      if(c.getSpecies()==Species.HAWK)g.hawks++; else g.doves++;
    }
    return g;
  }
}
"@ | Set-Content "$ENGINE/GameSnapshot.java"

# ===================== CONTROLLER =====================
@"
package com.game.gametheory.controller;

import com.game.gametheory.engine.*;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/game")
public class GameController {
  private GameEngine engine;

  @PostMapping("/start")
  public void start(@RequestParam int hawks,@RequestParam int doves){
    engine=new GameEngine(hawks,doves);
  }

  @GetMapping("/step")
  public GameSnapshot step(){
    return engine.nextDay();
  }
}
"@ | Set-Content "$CTRL/GameController.java"

# ===================== FRONTEND =====================
@"
<!DOCTYPE html>
<html>
<head>
<title>Game Theory</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
<button onclick="start()">Start</button>
<button onclick="step()">Next Day</button>
<canvas id="board" width="600" height="600"></canvas>
<canvas id="chart"></canvas>
<script src="hawk-dove.js"></script>
</body>
</html>
"@ | Set-Content "$STATIC/game.html"

@"
let ctx=document.getElementById('board').getContext('2d');
let chart=new Chart(document.getElementById('chart'),{
  type:'line',
  data:{labels:[],datasets:[
    {label:'Hawks',borderColor:'red',data:[]},
    {label:'Doves',borderColor:'blue',data:[]}
  ]}
});

function start(){
  fetch('/api/game/start?hawks=20&doves=20',{method:'POST'});
}

function step(){
  fetch('/api/game/step').then(r=>r.json()).then(draw);
}

function draw(s){
  ctx.clearRect(0,0,600,600);
  ctx.beginPath();
  ctx.arc(300,300,280,0,2*Math.PI);
  ctx.stroke();
  s.creatures.forEach(c=>{
    ctx.beginPath();
    ctx.arc(300+c.x,300+c.y,5,0,2*Math.PI);
    ctx.fillStyle=c.type==='HAWK'?'red':'blue';
    ctx.fill();
  });
  chart.data.labels.push(s.day);
  chart.data.datasets[0].data.push(s.hawks);
  chart.data.datasets[1].data.push(s.doves);
  chart.update();
}
"@ | Set-Content "$STATIC/hawk-dove.js"

Write-Host "PROJET COMPLET CREE - LANCEZ AVEC mvn spring-boot:run"

