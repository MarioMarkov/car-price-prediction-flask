brands = {"BMW": ["3", "1", "X1", "5", "7", "X5", "X3", "6", "2", "X6"], "Chevrolet": ["Captiva", "Nubira", "Aveo", "Matiz", "Spark", "Cruze", "Lacetti", "Kalos", "Camaro"], "Citroen": ["C4", "Grand", "C5", "C3", "Berlingo", "Xsara", "C1"], "Dacia": ["Sandero", "Dokker", "Duster", "Lodgy", "Logan"], "Fiat": ["Panda", "Sedici", "Bravo", "Idea", "Punto", "Ulysse", "Doblo", "Multipla", "500", "Grande", "Scudo", "Croma", "Stilo", "Qubo", "500L", "Fiorino"], "Ford": ["Fiesta", "Focus", "Fusion", "C-Max", "Mondeo", "Ranger", "Kuga", "Galaxy", "Connect", "S-Max"], "Honda": ["Jazz", "CR-V", "Accord", "Civic", "FR-V", "HR-V"], "Hyundai": ["i30", "Tucson", "ix35", "i40", "i10", "ix20", "i20", "Santa", "Getz", "Sonata", "Kona", "Grand"], "Jeep": ["Cherokee", "Grand", "Wrangler", "Compass"], "Kia": ["Ceed", "Rio", "Picanto", "Sorento", "Sportage", "Carens", "Soul", "Venga", "Carnival", "K"], "AlfaRomeo": ["156", "159", "MiTo", "GT", "147", "Giulietta"], "LandRover": ["Freelander", "Range", "Discovery", "Land"], "Mazda": ["CX", "3", "6", "2", "5", "B"], "Mercedes-Benz": ["ML", "C", "E", "CLS", "B", "A", "S", "CLK", "500"], "Mini": ["Countryman", "Clubman", "Cooper", "ONE"], "Mitsubishi": ["Pajero", "Space", "Colt", "Grandis", "ASX", "L200", "Outlander", "Lancer"], "Nissan": ["Juke", "Qashqai", "Micra", "Qashqai+2", "Note", "Navara", "Pathfinder", "Almera", "Primera", "Murano", "Terrano", "X-Trail"], "Opel": ["Antara", "Astra", "Meriva", "Corsa", "Insignia", "Zafira", "Vectra", "Mokka", "Combo"], "Peugeot": ["508", "407", "208", "207", "308", "307", "5008", "3008", "Partner", "206", "2008"], "Porsche": ["Cayenne", "Panamera", "Macan"], "Renault": ["Clio", "Megane", "Modus", "Kangoo", "Laguna", "Scenic", "Koleos", "Twingo", "Captur", "Espace", "Grand", "5"], "Seat": ["Ibiza", "Altea", "Leon", "Exeo", "Toledo", "Alhambra", "Cordoba"], "Skoda": ["Octavia", "Superb", "Roomster", "Fabia", "Yeti"], "Subaru": ["Impreza", "Legacy", "Forester", "OUTBACK", "Justy", "B9", "XV"], "Suzuki": ["SX4", "Swift", "Vitara", "Jimny", "Grand", "Liana", "Ignis"], "Audi": ["A4", "A3", "A6", "Q7", "A5", "Q5", "A8"], "Toyota": ["Corolla", "Avensis", "Auris", "Yaris", "RAV", "Land", "Aygo", "Verso"], "Volvo": ["V40", "V50", "V60", "S80", "V70", "S40", "XC", "C30", "S60"], "VW": ["Golf", "Passat", "Touran", "Touareg", "Tiguan", "Polo", "Caddy", "Sharan"]}

window.onload=function(){
  try {
    var e = document.getElementById("brand");
    var select = document.getElementById("model");
    let models = brands[e.value];
    //console.log(models)
  
    function removeOptions(selectElement) {
      var i, L = selectElement.options.length - 1;
        for(i = L; i >= 0; i--) {
          selectElement.remove(i);
        }
    }
    removeOptions(select);
    
    //Loop through the car models and add them as options to the select element
    for (const [key, value] of Object.entries(models)) {
      var option = document.createElement("option");
      option.text = value;
      select.add(option);
    }
    
    // Make get request when brand changes
    e.addEventListener("change", function() {
      let models = brands[e.value];
  
      function removeOptions(selectElement) {
        var i, L = selectElement.options.length - 1;
          for(i = L; i >= 0; i--) {
            selectElement.remove(i);
          }
      }
      removeOptions(select);
      
      //Loop through the car models and add them as options to the select element
      for (const [key, value] of Object.entries(models)) {
        var option = document.createElement("option");
        option.text = value;
        select.add(option);
      }
    
    });
  } catch (e) {
  }
 
};

setTimeout(function() {
  var alert = document.getElementById('alert-success');
  if (alert) {
      alert.style.display = 'none';
  }
}, 3000);



