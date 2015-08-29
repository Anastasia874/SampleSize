var express=require("express");
var multer  = require('multer');
var app=express();
var done=false; // !!!! Change to FALSE !!!
var PythonShell = require('python-shell');
var bodyParser = require('body-parser');

//var newFilename;

var path = require('path');

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({
  extended: true
}));
/*Configure the multer.*/

app.use(multer({ dest: './uploads/',
 
 rename: function (fieldname, filename) {
    vnewFilename = filename+Date.now();
    return 'file'; //filename+Date.now();
  },
onFileUploadStart: function (file) {
  console.log(file.originalname + ' is starting ...')
},
onFileUploadComplete: function (file) {
  console.log(file.fieldname + ' uploaded to  ' + file.path)
  done=true;
}
}));

/*Handling routes.*/

app.get('/',function(req,res){
      res.sendfile("index.html");
});

app.post('/api/photo',function(req,res){
    if(done==true){
    console.log(req.files);
    //res.send("File uploaded.");
      }
});

app.post('/api/par', function(req,res){
  var filename = 'file.txt';
  var msg = '';
  if(done==false){
    filename = 'test.txt';
    msg = 'No file uploaded. Processing test.txt\n';  
  }
  res.setHeader('Content-Type', 'text/plain')
  var alpha_par = req.body.alpha;
  var power_par = req.body.power;
  var method = req.body.method;
  var delta = 0;
  switch (method) {
    case 'Superiority test':
      console.log(method)
      delta = req.body.sup_delta;
    case 'Equivalence test':
      delta = req.body.eq_delta;
  }
  console.log(req.body)
  //if(done==true){
    console.log(req.files);
    var options = {
      mode: 'text',
      pythonPath: 'C:/Users/Anastasia/Anaconda/python',
      //pythonOptions: ['-u'],
      scriptPath: path.resolve(__dirname),//'C:/Users/Anastasia/Documents/Strijov/Sample size',
      args: [filename, method, alpha_par, power_par, delta]
    };
    PythonShell.run('sample_size.py', options, function (err, results) {
      if (err) throw err;
      //console.log('finished');
      console.log('results: %j', results);
      res.send(msg + results);
    });
  //}
});
/*Run the server.*/
app.listen(3000,function(){
    console.log("Working on port 3000");
});



 
//var shell = new PythonShell('my_script.py', options);
//shell.send('hello world!');




