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
      var actions = ["up", "down", "left", "right"];
  
      // 全域陣列：V 為狀態值，randomPolicy 為每個格子的隨機行動
      var V = [];
      var randomPolicy = [];
  
      // 初始化 V 與 randomPolicy：依照格子的狀態決定初始值
      function initPolicyEvaluationArrays() {
          V = [];
          randomPolicy = [];
          for (var i = 0; i < dimension; i++) {
              V.push([]);
              randomPolicy.push([]);
              for (var j = 0; j < dimension; j++) {
                  var cell = getCellElement(i, j);
                  // 障礙物與終點狀態固定，其他狀態初始值為 0 且隨機指派一個行動
                  if (cell.classList.contains("obstacle")) {
                      V[i].push(-1);
                      randomPolicy[i].push(null);
                  } else if (cell.classList.contains("end")) {
                      V[i].push(20);
                      randomPolicy[i].push(null);
                  } else {
                      V[i].push(0);
                      randomPolicy[i].push(actions[Math.floor(Math.random() * actions.length)]);
                  }
              }
          }
      }
  
      // 取得指定列、行的單元格元素
      function getCellElement(row, col) {
          return document.querySelector('td[data-row="'+ row +'"][data-col="'+ col +'"]');
      }
  
      // 將行動轉換成箭頭符號
      function actionToArrow(action) {
          if (action === "up") return "↑";
          if (action === "down") return "↓";
          if (action === "left") return "←";
          if (action === "right") return "→";
          return "";
      }
  
      // 給定當前狀態與行動，計算下一狀態
      // 若碰到邊界或障礙物則停留在原位
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
  
      // 政策評估：根據目前隨機政策迭代更新 V(s)
      function policyEvaluation() {
          for (var iter = 0; iter < maxIterations; iter++) {
              var delta = 0;
              var newV = [];
              for (var i = 0; i < dimension; i++) {
                  newV.push([]);
                  for (var j = 0; j < dimension; j++) {
                      var cell = getCellElement(i, j);
                      if (cell.classList.contains("obstacle")) {
                          newV[i].push(-1);
                      } else if (cell.classList.contains("end")) {
                          newV[i].push(20);
                      } else {
                          var action = randomPolicy[i][j];
                          var nextState = getNextState(i, j, action);
                          var reward = -1;
                          var value = reward + gamma * V[nextState.row][nextState.col];
                          newV[i].push(value);
                          delta = Math.max(delta, Math.abs(value - V[i][j]));
                      }
                  }
              }
              V = newV;
              if (delta < threshold) break;
          }
      }
  
      // 更新格子內容：同時顯示隨機政策箭頭與狀態價值 (V(s))
      function updatePolicyDisplay() {
          for (var i = 0; i < dimension; i++) {
              for (var j = 0; j < dimension; j++) {
                  var cell = getCellElement(i, j);
                  cell.innerHTML = "";  // 清除之前內容
                  // 若非障礙、非終點則顯示政策箭頭
                  if (!cell.classList.contains("obstacle") && !cell.classList.contains("end")) {
                      var arrowSpan = document.createElement("span");
                      arrowSpan.className = "policyArrow";
                      arrowSpan.textContent = actionToArrow(randomPolicy[i][j]);
                      cell.appendChild(arrowSpan);
                  }
                  // 顯示狀態價值，保留一位小數
                  var valueSpan = document.createElement("span");
                  valueSpan.className = "stateValue";
                  valueSpan.textContent = V[i][j].toFixed(1);
                  cell.appendChild(valueSpan);
              }
          }
      }
  
      // 主函式：初始化隨機政策與 V，執行政策評估，並更新顯示
      window.evaluatePolicy = function() {
          initPolicyEvaluationArrays();
          policyEvaluation();
          updatePolicyDisplay();
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
