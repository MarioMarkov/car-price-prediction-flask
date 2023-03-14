

window.onload=function(){
  var e = document.getElementById("brand");
  // Make get request when it first loads
  var value = e.value;
  var val = e.options[e.selectedIndex].text;
  var xhr = new XMLHttpRequest();
  xhr.open("GET", `http://127.0.0.1:5000/get_brands/${val}`);
  xhr.send();
  
   //Handle the response
   xhr.onload = function() {
      if (xhr.status === 200) {
        //Parse the JSON response
        var carModels = JSON.parse(xhr.responseText).models;
        //Get the select element
        var select = document.getElementById("model");
        function removeOptions(selectElement) {
            var i, L = selectElement.options.length - 1;
            for(i = L; i >= 0; i--) {
                selectElement.remove(i);
            }
          }
        removeOptions(select);
  
  
        //Loop through the car models and add them as options to the select element
        for (var i = 0; i < carModels.length; i++) {
          var option = document.createElement("option");
          option.text = carModels[i];
          
          select.add(option);
        }
  
      } else {
        console.log("Error retrieving car models");
      }
    };
  
  // Make get request when brand changes
  e.addEventListener("change", function() {
    var value = e.value;
    var val = e.options[e.selectedIndex].text;
    var xhr = new XMLHttpRequest();
    xhr.open("GET", `http://127.0.0.1:5000/get_brands/${val}`);
    xhr.send();
  
    //Handle the response
    xhr.onload = function() {
      if (xhr.status === 200) {
        //Parse the JSON response
        var carModels = JSON.parse(xhr.responseText).models;
        //Get the select element
        var select = document.getElementById("model");
        function removeOptions(selectElement) {
            var i, L = selectElement.options.length - 1;
            for(i = L; i >= 0; i--) {
                selectElement.remove(i);
            }
          }
        removeOptions(select);
  
  
        //Loop through the car models and add them as options to the select element
        for (var i = 0; i < carModels.length; i++) {
          var option = document.createElement("option");
          option.text = carModels[i];
          
          select.add(option);
        }
  
      } else {
        console.log("Error retrieving car models");
      }
    };
  });

  
  
  // Set hidden input to a value 
  ml_model_input = document.getElementById("hidden_model_input")
  dt_model = document.getElementById("dt_model")
  nn_model = document.getElementById("nn_model")
  if(dt_model.checked){
    ml_model_input.value = "dt"
  }else{
    ml_model_input.value = "nn"
  }
  console.log(ml_model_input.value)
  var radios = document.forms["model_select"].elements["ml_model"];
  for(var i = 0, max = radios.length; i < max; i++) {
      radios[i].onclick = function() {
        ml_model_input.value = this.value
        console.log(ml_model_input.value)
      }
  }
};

