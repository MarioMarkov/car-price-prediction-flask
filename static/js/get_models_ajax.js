window.onload=function(){
  var e = document.getElementById("brand");
  // Make get request when it first loads
  var value = e.value;
  var val = e.options[e.selectedIndex].text;
  var xhr = new XMLHttpRequest();
  xhr.open("GET", `https://carspredictweb-production.up.railway.app/get_brands/${val}`);
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
    xhr.open("GET", `https://carspredictweb-production.up.railway.app/get_brands/${val}`);
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

};

setTimeout(function() {
  var alert = document.getElementById('alert-success');
  if (alert) {
      alert.style.display = 'none';
  }
}, 3000);



