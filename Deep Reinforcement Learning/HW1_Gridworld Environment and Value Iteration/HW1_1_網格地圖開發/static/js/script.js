(function() {
    // 當文件載入完畢後執行
    document.addEventListener('DOMContentLoaded', function(){
        var gridTable = document.getElementById("gridTable");
        if (!gridTable) return;  // 若網格尚未生成則跳出

        var dimension = parseInt(gridTable.getAttribute("data-dimension"));
        var maxObstacles = dimension - 2;
        var currentObstacleCount = 0;

        // 更新障礙物計數顯示
        function updateObstacleDisplay() {
            var display = document.getElementById("obstacleCountDisplay");
            if (display) {
                display.textContent = "障礙物: " + currentObstacleCount + " / " + maxObstacles;
            }
        }

        // 設定單一格子狀態
        window.setCell = function(cell) {
            var mode = document.querySelector('input[name="mode"]:checked').value;
            
            if (mode === "start") {
                // 若點選的格子已是結束或障礙物則不允許
                if (cell.classList.contains("end") || cell.classList.contains("obstacle")) {
                    alert("無法將起始單元格設在已指定的障礙物或結束單元格上！");
                    return;
                }
                // 移除原有起始單元格（若存在）
                var currentStart = document.querySelector("td.start");
                if (currentStart) {
                    currentStart.classList.remove("start");
                    currentStart.classList.add("empty");
                }
                cell.classList.remove("empty");
                cell.classList.add("start");
            } else if (mode === "end") {
                // 若點選的格子已是起始或障礙物則不允許
                if (cell.classList.contains("start") || cell.classList.contains("obstacle")) {
                    alert("無法將結束單元格設在起始單元格或障礙物上！");
                    return;
                }
                // 移除原有結束單元格（若存在）
                var currentEnd = document.querySelector("td.end");
                if (currentEnd) {
                    currentEnd.classList.remove("end");
                    currentEnd.classList.add("empty");
                }
                cell.classList.remove("empty");
                cell.classList.add("end");
            } else if (mode === "obstacle") {
                // 若格子已為起始或結束則不允許變更為障礙物
                if (cell.classList.contains("start") || cell.classList.contains("end")) {
                    alert("無法將障礙物設在起始或結束單元格上！");
                    return;
                }
                // 若已為障礙物則點擊可移除，否則新增障礙物（若未達上限）
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
                updateObstacleDisplay();
            }
        }

        // 重設網格，將所有格子回復至 empty 狀態，並重置障礙物計數
        window.resetGrid = function() {
            var cells = document.querySelectorAll("#gridTable td");
            cells.forEach(function(cell) {
                cell.className = "empty";
            });
            currentObstacleCount = 0;
            updateObstacleDisplay();
        }

        // 初始化障礙物計數顯示
        updateObstacleDisplay();
    });
})();
