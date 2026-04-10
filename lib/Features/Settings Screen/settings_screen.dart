import 'package:flutter/material.dart';

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  

  @override
  Widget build(BuildContext context) {

List<String> myList = [
    'You can change the model',
    'the confidence threshold',
    'the tracking threshold',
    'the prediction frames',
    'the history size',
    'the collision distance',
    'the TTC threshold',
    'the colors',
    'the model name'
  ];

    return Scaffold(
      body: SingleChildScrollView(
        child: Padding(
          padding: EdgeInsets.all(16.0),
          child: Column(
          children: [
            Text('All Configurables parameters will be listed here.', style: TextStyle(fontSize: 20.0, fontWeight: FontWeight.bold)),
            ListView.builder(
              shrinkWrap: true,
              itemCount: myList.length,
              itemBuilder: (context, index) {
                return Text('• ${myList[index]}', style: TextStyle(fontSize: 16.0));
              },
            ),  
          ],
        )
      )
    )
  );
}
}
