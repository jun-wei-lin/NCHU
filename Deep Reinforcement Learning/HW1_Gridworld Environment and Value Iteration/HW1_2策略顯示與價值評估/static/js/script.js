(function() {
    document.addEventListener('DOMContentLoaded', function(){
      var gridTable = document.getElementById("gridTable");
      if (!gridTable) return;
  
      var dimension = parseInt(gridTable.getAttribute("data-dimension"));
      var maxObstacles = dimension - 2;
      var currentObstacleCount = 0;
      var gamma = 0.9;
      var threshold = 0.001;
      var maxIterations = 100;
  
      // 全域陣列：最優價值 V 與最優策略
      var V = [];
      var optimalPolicy = [];
  
      // 初始化 V 與 optimalPolicy
      function initOptimalArrays() {
        V = [];
        optimalPolicy = [];
        for (var i = 0; i < dimension; i++) {
          V.push([]);
          optimalPolicy.push([]);
          for (var j = 0; j < dimension; j++) {
            V[i].push(0);
            optimalPolicy[i].push(null);
          }
        }
      }
  
      // 取得單元格元素
      function getCellElement(row, col) {
        return document.querySelector('td[data-row="'+ row +'"][data-col="'+ col +'"]');
      }
  
      // 根據動作計算下一狀態（遇邊界或障礙物則停留）
      function getNextState(row, col, action) {
        var nextRow = row, nextCol = col;
        if (action === "up") nextRow = row - 1;
        else if (action === "down") nextRow = row + 1;
        else if (action === "left") nextCol = col - 1;
        else if (action === "right") nextCol = col + 1;
        if (nextRow < 0 || nextRow >= dimension || nextCol < 0 || nextCol >= dimension) {
          return {row: row, col: col};
        }
        var nextCell = getCellElement(nextRow, nextCol);
        if (nextCell && nextCell.classList.contains("obstacle")) {
          return {row: row, col: col};
        }
        return {row: nextRow, col: nextCol};
      }
  
      // 將行動轉為箭頭符號
      function actionToArrow(action) {
        if (action === "up") return "↑";
        if (action === "down") return "↓";
        if (action === "left") return "←";
        if (action === "right") return "→";
        return "";
      }
  
      // Value Iteration：對每個非障礙、非終點狀態更新價值
      function valueIteration() {
        for (var iter = 0; iter < maxIterations; iter++) {
          var delta = 0;
          var newV = [];
          for (var i = 0; i < dimension; i++) {
            newV.push([]);
            for (var j = 0; j < dimension; j++) {
              var cell = getCellElement(i, j);
              // 障礙物固定值 -1；終點固定值 20
              if (cell.classList.contains("obstacle")) {
                newV[i][j] = -1;
              } else if (cell.classList.contains("end")) {
                newV[i][j] = 20;
              } else {
                var bestValue = -Infinity;
                var actions = ["up", "down", "left", "right"];
                for (var a = 0; a < actions.length; a++) {
                  var action = actions[a];
                  var nextState = getNextState(i, j, action);
                  var reward = -1; // 每步 -1
                  var value = reward + gamma * V[nextState.row][nextState.col];
                  if (value > bestValue) bestValue = value;
                }
                newV[i][j] = bestValue;
                delta = Math.max(delta, Math.abs(newV[i][j] - V[i][j]));
              }
            }
          }
          V = newV;
          if (delta < threshold) break;
        }
      }
  
      // 根據 V 提取最優策略：在每個狀態選擇使 reward + γ * V(next) 最大的行動
      function extractOptimalPolicy() {
        var actions = ["up", "down", "left", "right"];
        for (var i = 0; i < dimension; i++) {
          for (var j = 0; j < dimension; j++) {
            var cell = getCellElement(i, j);
            if (cell.classList.contains("obstacle") || cell.classList.contains("end")) {
              optimalPolicy[i][j] = null;
            } else {
              var bestAction = null;
              var bestValue = -Infinity;
              for (var a = 0; a < actions.length; a++) {
                var action = actions[a];
                var nextState = getNextState(i, j, action);
                var reward = -1;
                var value = reward + gamma * V[nextState.row][nextState.col];
                if (value > bestValue) {
                  bestValue = value;
                  bestAction = action;
                }
              }
              optimalPolicy[i][j] = bestAction;
            }
          }
        }
      }
  
      // 將每個單元格上顯示最優策略箭頭與狀態價值（保留一位小數）
      function updateOptimalDisplay() {
        for (var i = 0; i < dimension; i++) {
          for (var j = 0; j < dimension; j++) {
            var cell = getCellElement(i, j);
            cell.innerHTML = "";
            if (optimalPolicy[i][j]) {
              var arrowSpan = document.createElement("span");
              arrowSpan.className = "policyArrow";
              arrowSpan.textContent = actionToArrow(optimalPolicy[i][j]);
              cell.appendChild(arrowSpan);
            }
            var valueSpan = document.createElement("span");
            valueSpan.className = "stateValue";
            valueSpan.textContent = V[i][j].toFixed(1);
            cell.appendChild(valueSpan);
          }
        }
      }
  
      // 從起始單元格依據 optimalPolicy 模擬最佳路徑並高亮顯示
      function simulateOptimalPath() {
        var startCell = document.querySelector("td.start");
        if (!startCell) {
          alert("請先設定起始單元格！");
          return;
        }
        var row = parseInt(startCell.getAttribute("data-row"));
        var col = parseInt(startCell.getAttribute("data-col"));
        var path = [];
        var visited = new Set();
        while (true) {
          var cell = getCellElement(row, col);
          path.push({row: row, col: col});
          // 如果到達終點則結束
          if (cell.classList.contains("end")) break;
          var key = row + "," + col;
          if (visited.has(key)) {
            alert("發現循環，無法到達終點！");
            break;
          }
          visited.add(key);
          var action = optimalPolicy[row][col];
          if (!action) {
            alert("無法找到最佳行動！");
            break;
          }
          var nextState = getNextState(row, col, action);
          // 若無法前進，跳出迴圈
          if (nextState.row === row && nextState.col === col) {
            alert("無法前進！");
            break;
          }
          row = nextState.row;
          col = nextState.col;
        }
        // 高亮路徑：在每個屬於最佳路徑的單元格加上 .agent 樣式
        path.forEach(function(pos) {
          var cell = getCellElement(pos.row, pos.col);
          cell.classList.add("agent");
        });
      }
  
      // 主函式：初始化、計算最優價值與策略，更新顯示，並模擬最佳路徑
      window.findOptimalPath = function() {
        initOptimalArrays();
        valueIteration();
        extractOptimalPolicy();
        updateOptimalDisplay();
        simulateOptimalPath();
      };
  
      /* ---------------------------
         以下為原有設定單元格與重設功能
      --------------------------- */
      window.setCell = function(cell) {
        var mode = document.querySelector('input[name="mode"]:checked').value;
        if (mode === "start") {
          if (cell.classList.contains("end") || cell.classList.contains("obstacle")) {
            alert("無法將起始單元格設在已指定的障礙物或結束單元格上！");
            return;
          }
          var currentStart = document.querySelector("td.start");
          if (currentStart) {
            currentStart.classList.remove("start");
            currentStart.classList.add("empty");
          }
          cell.classList.remove("empty");
          cell.classList.add("start");
        } else if (mode === "end") {
          if (cell.classList.contains("start") || cell.classList.contains("obstacle")) {
            alert("無法將結束單元格設在起始單元格或障礙物上！");
            return;
          }
          var currentEnd = document.querySelector("td.end");
          if (currentEnd) {
            currentEnd.classList.remove("end");
            currentEnd.classList.add("empty");
          }
          cell.classList.remove("empty");
          cell.classList.add("end");
        } else if (mode === "obstacle") {
          if (cell.classList.contains("start") || cell.classList.contains("end")) {
            alert("無法將障礙物設在起始或結束單元格上！");
            return;
          }
          if (cell.classList.contains("obstacle")) {
            cell.classList.remove("obstacle");
            cell.classList.add("empty");
            currentObstacleCount--;
          } else {
            if (currentObstacleCount < maxObstacles) {
              cell.classList.remove("empty");
              cell.classList.add("obstacle");
              currentObstacleCount++;
            } else {
              alert("障礙物數量已達上限！");
            }
          }
          var display = document.getElementById("obstacleCountDisplay");
          if (display) {
            display.textContent = "障礙物: " + currentObstacleCount + " / " + maxObstacles;
          }
        }
      };
  
      window.resetGrid = function() {
        var cells = document.querySelectorAll("#gridTable td");
        cells.forEach(function(cell) {
          cell.className = "empty";
          cell.innerHTML = "";
        });
        currentObstacleCount = 0;
        var display = document.getElementById("obstacleCountDisplay");
        if (display) {
          display.textContent = "障礙物: 0 / " + maxObstacles;
        }
      };
    });
  })();
  